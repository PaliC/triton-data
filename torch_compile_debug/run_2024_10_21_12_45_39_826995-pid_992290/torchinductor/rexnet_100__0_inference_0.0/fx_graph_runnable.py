
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1):
        convolution_75 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_135 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_62 = torch.ops.aten.sqrt.default(add_135);  add_135 = None
        reciprocal_62 = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_216 = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62 = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_497);  convolution_75 = unsqueeze_497 = None
        mul_217 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_218 = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_501);  mul_217 = unsqueeze_501 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_136 = torch.ops.aten.add.Tensor(mul_218, unsqueeze_503);  mul_218 = unsqueeze_503 = None
        sigmoid_30 = torch.ops.aten.sigmoid.default(add_136)
        mul_219 = torch.ops.aten.mul.Tensor(add_136, sigmoid_30);  add_136 = sigmoid_30 = None
        convolution_76 = torch.ops.aten.convolution.default(mul_219, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  mul_219 = arg6_1 = None
        add_137 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_63 = torch.ops.aten.sqrt.default(add_137);  add_137 = None
        reciprocal_63 = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_220 = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(mul_220, -1);  mul_220 = None
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63 = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_505);  convolution_76 = unsqueeze_505 = None
        mul_221 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_222 = torch.ops.aten.mul.Tensor(mul_221, unsqueeze_509);  mul_221 = unsqueeze_509 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_138 = torch.ops.aten.add.Tensor(mul_222, unsqueeze_511);  mul_222 = unsqueeze_511 = None
        clamp_min_16 = torch.ops.aten.clamp_min.default(add_138, 0.0);  add_138 = None
        clamp_max_16 = torch.ops.aten.clamp_max.default(clamp_min_16, 6.0);  clamp_min_16 = None
        convolution_77 = torch.ops.aten.convolution.default(clamp_max_16, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_16 = arg11_1 = None
        add_139 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_64 = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_64 = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_223 = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(mul_223, -1);  mul_223 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64 = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_513);  convolution_77 = unsqueeze_513 = None
        mul_224 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_517);  mul_224 = unsqueeze_517 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_140 = torch.ops.aten.add.Tensor(mul_225, unsqueeze_519);  mul_225 = unsqueeze_519 = None
        convolution_78 = torch.ops.aten.convolution.default(add_140, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_140 = arg16_1 = None
        add_141 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_65 = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_65 = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_226 = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(mul_226, -1);  mul_226 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65 = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_521);  convolution_78 = unsqueeze_521 = None
        mul_227 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_228 = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_525);  mul_227 = unsqueeze_525 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_142 = torch.ops.aten.add.Tensor(mul_228, unsqueeze_527);  mul_228 = unsqueeze_527 = None
        sigmoid_31 = torch.ops.aten.sigmoid.default(add_142)
        mul_229 = torch.ops.aten.mul.Tensor(add_142, sigmoid_31);  add_142 = sigmoid_31 = None
        convolution_79 = torch.ops.aten.convolution.default(mul_229, arg21_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96);  mul_229 = arg21_1 = None
        add_143 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_66 = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_66 = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_230 = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(mul_230, -1);  mul_230 = None
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66 = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_529);  convolution_79 = unsqueeze_529 = None
        mul_231 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_533);  mul_231 = unsqueeze_533 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_144 = torch.ops.aten.add.Tensor(mul_232, unsqueeze_535);  mul_232 = unsqueeze_535 = None
        clamp_min_17 = torch.ops.aten.clamp_min.default(add_144, 0.0);  add_144 = None
        clamp_max_17 = torch.ops.aten.clamp_max.default(clamp_min_17, 6.0);  clamp_min_17 = None
        convolution_80 = torch.ops.aten.convolution.default(clamp_max_17, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_17 = arg26_1 = None
        add_145 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_67 = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_67 = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_233 = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(mul_233, -1);  mul_233 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_537);  convolution_80 = unsqueeze_537 = None
        mul_234 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_541);  mul_234 = unsqueeze_541 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_543 = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_146 = torch.ops.aten.add.Tensor(mul_235, unsqueeze_543);  mul_235 = unsqueeze_543 = None
        convolution_81 = torch.ops.aten.convolution.default(add_146, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg31_1 = None
        add_147 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_68 = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_68 = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_236 = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_545 = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(mul_236, -1);  mul_236 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68 = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_545);  convolution_81 = unsqueeze_545 = None
        mul_237 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_238 = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_549);  mul_237 = unsqueeze_549 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_148 = torch.ops.aten.add.Tensor(mul_238, unsqueeze_551);  mul_238 = unsqueeze_551 = None
        sigmoid_32 = torch.ops.aten.sigmoid.default(add_148)
        mul_239 = torch.ops.aten.mul.Tensor(add_148, sigmoid_32);  add_148 = sigmoid_32 = None
        convolution_82 = torch.ops.aten.convolution.default(mul_239, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 162);  mul_239 = arg36_1 = None
        add_149 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_69 = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_69 = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_240 = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69 = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_553);  convolution_82 = unsqueeze_553 = None
        mul_241 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_557);  mul_241 = unsqueeze_557 = None
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_150 = torch.ops.aten.add.Tensor(mul_242, unsqueeze_559);  mul_242 = unsqueeze_559 = None
        clamp_min_18 = torch.ops.aten.clamp_min.default(add_150, 0.0);  add_150 = None
        clamp_max_18 = torch.ops.aten.clamp_max.default(clamp_min_18, 6.0);  clamp_min_18 = None
        convolution_83 = torch.ops.aten.convolution.default(clamp_max_18, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_18 = arg41_1 = None
        add_151 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_70 = torch.ops.aten.sqrt.default(add_151);  add_151 = None
        reciprocal_70 = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_243 = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70 = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_561);  convolution_83 = unsqueeze_561 = None
        mul_244 = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_565);  mul_244 = unsqueeze_565 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_152 = torch.ops.aten.add.Tensor(mul_245, unsqueeze_567);  mul_245 = unsqueeze_567 = None
        slice_46 = torch.ops.aten.slice.Tensor(add_152, 1, 0, 27)
        add_153 = torch.ops.aten.add.Tensor(slice_46, add_146);  slice_46 = add_146 = None
        slice_48 = torch.ops.aten.slice.Tensor(add_152, 1, 27, 9223372036854775807);  add_152 = None
        cat_11 = torch.ops.aten.cat.default([add_153, slice_48], 1);  add_153 = slice_48 = None
        convolution_84 = torch.ops.aten.convolution.default(cat_11, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_11 = arg46_1 = None
        add_154 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_71 = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_71 = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_246 = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71 = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_569);  convolution_84 = unsqueeze_569 = None
        mul_247 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_573);  mul_247 = unsqueeze_573 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_155 = torch.ops.aten.add.Tensor(mul_248, unsqueeze_575);  mul_248 = unsqueeze_575 = None
        sigmoid_33 = torch.ops.aten.sigmoid.default(add_155)
        mul_249 = torch.ops.aten.mul.Tensor(add_155, sigmoid_33);  add_155 = sigmoid_33 = None
        convolution_85 = torch.ops.aten.convolution.default(mul_249, arg51_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 228);  mul_249 = arg51_1 = None
        add_156 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_72 = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_72 = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_250 = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72 = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_577);  convolution_85 = unsqueeze_577 = None
        mul_251 = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_581);  mul_251 = unsqueeze_581 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_157 = torch.ops.aten.add.Tensor(mul_252, unsqueeze_583);  mul_252 = unsqueeze_583 = None
        mean_14 = torch.ops.aten.mean.dim(add_157, [2, 3], True)
        convolution_86 = torch.ops.aten.convolution.default(mean_14, arg56_1, arg57_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg56_1 = arg57_1 = None
        add_158 = torch.ops.aten.add.Tensor(arg59_1, 1e-05);  arg59_1 = None
        sqrt_73 = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_73 = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_253 = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73 = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_585);  convolution_86 = unsqueeze_585 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_589);  mul_254 = unsqueeze_589 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_159 = torch.ops.aten.add.Tensor(mul_255, unsqueeze_591);  mul_255 = unsqueeze_591 = None
        relu_13 = torch.ops.aten.relu.default(add_159);  add_159 = None
        convolution_87 = torch.ops.aten.convolution.default(relu_13, arg62_1, arg63_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg62_1 = arg63_1 = None
        sigmoid_34 = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None
        mul_256 = torch.ops.aten.mul.Tensor(add_157, sigmoid_34);  add_157 = sigmoid_34 = None
        clamp_min_19 = torch.ops.aten.clamp_min.default(mul_256, 0.0);  mul_256 = None
        clamp_max_19 = torch.ops.aten.clamp_max.default(clamp_min_19, 6.0);  clamp_min_19 = None
        convolution_88 = torch.ops.aten.convolution.default(clamp_max_19, arg64_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_19 = arg64_1 = None
        add_160 = torch.ops.aten.add.Tensor(arg66_1, 1e-05);  arg66_1 = None
        sqrt_74 = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_74 = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_257 = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(mul_257, -1);  mul_257 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74 = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_593);  convolution_88 = unsqueeze_593 = None
        mul_258 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_597);  mul_258 = unsqueeze_597 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_161 = torch.ops.aten.add.Tensor(mul_259, unsqueeze_599);  mul_259 = unsqueeze_599 = None
        convolution_89 = torch.ops.aten.convolution.default(add_161, arg69_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg69_1 = None
        add_162 = torch.ops.aten.add.Tensor(arg71_1, 1e-05);  arg71_1 = None
        sqrt_75 = torch.ops.aten.sqrt.default(add_162);  add_162 = None
        reciprocal_75 = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_260 = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(mul_260, -1);  mul_260 = None
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75 = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_601);  convolution_89 = unsqueeze_601 = None
        mul_261 = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_605 = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_262 = torch.ops.aten.mul.Tensor(mul_261, unsqueeze_605);  mul_261 = unsqueeze_605 = None
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_163 = torch.ops.aten.add.Tensor(mul_262, unsqueeze_607);  mul_262 = unsqueeze_607 = None
        sigmoid_35 = torch.ops.aten.sigmoid.default(add_163)
        mul_263 = torch.ops.aten.mul.Tensor(add_163, sigmoid_35);  add_163 = sigmoid_35 = None
        convolution_90 = torch.ops.aten.convolution.default(mul_263, arg74_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 300);  mul_263 = arg74_1 = None
        add_164 = torch.ops.aten.add.Tensor(arg76_1, 1e-05);  arg76_1 = None
        sqrt_76 = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_76 = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_264 = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_611 = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76 = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_609);  convolution_90 = unsqueeze_609 = None
        mul_265 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_613 = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_613);  mul_265 = unsqueeze_613 = None
        unsqueeze_614 = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_615 = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_165 = torch.ops.aten.add.Tensor(mul_266, unsqueeze_615);  mul_266 = unsqueeze_615 = None
        mean_15 = torch.ops.aten.mean.dim(add_165, [2, 3], True)
        convolution_91 = torch.ops.aten.convolution.default(mean_15, arg79_1, arg80_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg79_1 = arg80_1 = None
        add_166 = torch.ops.aten.add.Tensor(arg82_1, 1e-05);  arg82_1 = None
        sqrt_77 = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_77 = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_267 = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616 = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_617 = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618 = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_619 = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77 = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_617);  convolution_91 = unsqueeze_617 = None
        mul_268 = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_621);  mul_268 = unsqueeze_621 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_623 = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_167 = torch.ops.aten.add.Tensor(mul_269, unsqueeze_623);  mul_269 = unsqueeze_623 = None
        relu_14 = torch.ops.aten.relu.default(add_167);  add_167 = None
        convolution_92 = torch.ops.aten.convolution.default(relu_14, arg85_1, arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg85_1 = arg86_1 = None
        sigmoid_36 = torch.ops.aten.sigmoid.default(convolution_92);  convolution_92 = None
        mul_270 = torch.ops.aten.mul.Tensor(add_165, sigmoid_36);  add_165 = sigmoid_36 = None
        clamp_min_20 = torch.ops.aten.clamp_min.default(mul_270, 0.0);  mul_270 = None
        clamp_max_20 = torch.ops.aten.clamp_max.default(clamp_min_20, 6.0);  clamp_min_20 = None
        convolution_93 = torch.ops.aten.convolution.default(clamp_max_20, arg87_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_20 = arg87_1 = None
        add_168 = torch.ops.aten.add.Tensor(arg89_1, 1e-05);  arg89_1 = None
        sqrt_78 = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_78 = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_271 = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624 = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_625 = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626 = torch.ops.aten.unsqueeze.default(mul_271, -1);  mul_271 = None
        unsqueeze_627 = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78 = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_625);  convolution_93 = unsqueeze_625 = None
        mul_272 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_629 = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_272, unsqueeze_629);  mul_272 = unsqueeze_629 = None
        unsqueeze_630 = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_631 = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_169 = torch.ops.aten.add.Tensor(mul_273, unsqueeze_631);  mul_273 = unsqueeze_631 = None
        slice_50 = torch.ops.aten.slice.Tensor(add_169, 1, 0, 50)
        add_170 = torch.ops.aten.add.Tensor(slice_50, add_161);  slice_50 = add_161 = None
        slice_52 = torch.ops.aten.slice.Tensor(add_169, 1, 50, 9223372036854775807);  add_169 = None
        cat_12 = torch.ops.aten.cat.default([add_170, slice_52], 1);  add_170 = slice_52 = None
        convolution_94 = torch.ops.aten.convolution.default(cat_12, arg92_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_12 = arg92_1 = None
        add_171 = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
        sqrt_79 = torch.ops.aten.sqrt.default(add_171);  add_171 = None
        reciprocal_79 = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_274 = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(mul_274, -1);  mul_274 = None
        unsqueeze_635 = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79 = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_633);  convolution_94 = unsqueeze_633 = None
        mul_275 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_637 = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_275, unsqueeze_637);  mul_275 = unsqueeze_637 = None
        unsqueeze_638 = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_639 = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_172 = torch.ops.aten.add.Tensor(mul_276, unsqueeze_639);  mul_276 = unsqueeze_639 = None
        sigmoid_37 = torch.ops.aten.sigmoid.default(add_172)
        mul_277 = torch.ops.aten.mul.Tensor(add_172, sigmoid_37);  add_172 = sigmoid_37 = None
        convolution_95 = torch.ops.aten.convolution.default(mul_277, arg97_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 366);  mul_277 = arg97_1 = None
        add_173 = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
        sqrt_80 = torch.ops.aten.sqrt.default(add_173);  add_173 = None
        reciprocal_80 = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_278 = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640 = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_641 = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(mul_278, -1);  mul_278 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80 = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_641);  convolution_95 = unsqueeze_641 = None
        mul_279 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_280 = torch.ops.aten.mul.Tensor(mul_279, unsqueeze_645);  mul_279 = unsqueeze_645 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_174 = torch.ops.aten.add.Tensor(mul_280, unsqueeze_647);  mul_280 = unsqueeze_647 = None
        mean_16 = torch.ops.aten.mean.dim(add_174, [2, 3], True)
        convolution_96 = torch.ops.aten.convolution.default(mean_16, arg102_1, arg103_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_16 = arg102_1 = arg103_1 = None
        add_175 = torch.ops.aten.add.Tensor(arg105_1, 1e-05);  arg105_1 = None
        sqrt_81 = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_81 = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_281 = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(mul_281, -1);  mul_281 = None
        unsqueeze_651 = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81 = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_649);  convolution_96 = unsqueeze_649 = None
        mul_282 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652 = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_653 = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_283 = torch.ops.aten.mul.Tensor(mul_282, unsqueeze_653);  mul_282 = unsqueeze_653 = None
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_176 = torch.ops.aten.add.Tensor(mul_283, unsqueeze_655);  mul_283 = unsqueeze_655 = None
        relu_15 = torch.ops.aten.relu.default(add_176);  add_176 = None
        convolution_97 = torch.ops.aten.convolution.default(relu_15, arg108_1, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg108_1 = arg109_1 = None
        sigmoid_38 = torch.ops.aten.sigmoid.default(convolution_97);  convolution_97 = None
        mul_284 = torch.ops.aten.mul.Tensor(add_174, sigmoid_38);  add_174 = sigmoid_38 = None
        clamp_min_21 = torch.ops.aten.clamp_min.default(mul_284, 0.0);  mul_284 = None
        clamp_max_21 = torch.ops.aten.clamp_max.default(clamp_min_21, 6.0);  clamp_min_21 = None
        convolution_98 = torch.ops.aten.convolution.default(clamp_max_21, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_21 = arg110_1 = None
        add_177 = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_82 = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_82 = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_285 = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
        unsqueeze_659 = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82 = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_657);  convolution_98 = unsqueeze_657 = None
        mul_286 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660 = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_661 = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_661);  mul_286 = unsqueeze_661 = None
        unsqueeze_662 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_663 = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_178 = torch.ops.aten.add.Tensor(mul_287, unsqueeze_663);  mul_287 = unsqueeze_663 = None
        convolution_99 = torch.ops.aten.convolution.default(add_178, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg115_1 = None
        add_179 = torch.ops.aten.add.Tensor(arg117_1, 1e-05);  arg117_1 = None
        sqrt_83 = torch.ops.aten.sqrt.default(add_179);  add_179 = None
        reciprocal_83 = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_288 = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664 = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_665 = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666 = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_667 = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83 = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_665);  convolution_99 = unsqueeze_665 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_669);  mul_289 = unsqueeze_669 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_671 = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_180 = torch.ops.aten.add.Tensor(mul_290, unsqueeze_671);  mul_290 = unsqueeze_671 = None
        sigmoid_39 = torch.ops.aten.sigmoid.default(add_180)
        mul_291 = torch.ops.aten.mul.Tensor(add_180, sigmoid_39);  add_180 = sigmoid_39 = None
        convolution_100 = torch.ops.aten.convolution.default(mul_291, arg120_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 432);  mul_291 = arg120_1 = None
        add_181 = torch.ops.aten.add.Tensor(arg122_1, 1e-05);  arg122_1 = None
        sqrt_84 = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_84 = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_292 = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_673 = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674 = torch.ops.aten.unsqueeze.default(mul_292, -1);  mul_292 = None
        unsqueeze_675 = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84 = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_673);  convolution_100 = unsqueeze_673 = None
        mul_293 = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_677 = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_294 = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_677);  mul_293 = unsqueeze_677 = None
        unsqueeze_678 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_679 = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_182 = torch.ops.aten.add.Tensor(mul_294, unsqueeze_679);  mul_294 = unsqueeze_679 = None
        mean_17 = torch.ops.aten.mean.dim(add_182, [2, 3], True)
        convolution_101 = torch.ops.aten.convolution.default(mean_17, arg125_1, arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg125_1 = arg126_1 = None
        add_183 = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_85 = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_85 = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_295 = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85 = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_681);  convolution_101 = unsqueeze_681 = None
        mul_296 = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_685);  mul_296 = unsqueeze_685 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_687 = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_184 = torch.ops.aten.add.Tensor(mul_297, unsqueeze_687);  mul_297 = unsqueeze_687 = None
        relu_16 = torch.ops.aten.relu.default(add_184);  add_184 = None
        convolution_102 = torch.ops.aten.convolution.default(relu_16, arg131_1, arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_16 = arg131_1 = arg132_1 = None
        sigmoid_40 = torch.ops.aten.sigmoid.default(convolution_102);  convolution_102 = None
        mul_298 = torch.ops.aten.mul.Tensor(add_182, sigmoid_40);  add_182 = sigmoid_40 = None
        clamp_min_22 = torch.ops.aten.clamp_min.default(mul_298, 0.0);  mul_298 = None
        clamp_max_22 = torch.ops.aten.clamp_max.default(clamp_min_22, 6.0);  clamp_min_22 = None
        convolution_103 = torch.ops.aten.convolution.default(clamp_max_22, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_22 = arg133_1 = None
        add_185 = torch.ops.aten.add.Tensor(arg135_1, 1e-05);  arg135_1 = None
        sqrt_86 = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_86 = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_299 = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_689 = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(mul_299, -1);  mul_299 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86 = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_689);  convolution_103 = unsqueeze_689 = None
        mul_300 = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_301 = torch.ops.aten.mul.Tensor(mul_300, unsqueeze_693);  mul_300 = unsqueeze_693 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_186 = torch.ops.aten.add.Tensor(mul_301, unsqueeze_695);  mul_301 = unsqueeze_695 = None
        slice_54 = torch.ops.aten.slice.Tensor(add_186, 1, 0, 72)
        add_187 = torch.ops.aten.add.Tensor(slice_54, add_178);  slice_54 = add_178 = None
        slice_56 = torch.ops.aten.slice.Tensor(add_186, 1, 72, 9223372036854775807);  add_186 = None
        cat_13 = torch.ops.aten.cat.default([add_187, slice_56], 1);  add_187 = slice_56 = None
        convolution_104 = torch.ops.aten.convolution.default(cat_13, arg138_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg138_1 = None
        add_188 = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_87 = torch.ops.aten.sqrt.default(add_188);  add_188 = None
        reciprocal_87 = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_302 = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(mul_302, -1);  mul_302 = None
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87 = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_697);  convolution_104 = unsqueeze_697 = None
        mul_303 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_304 = torch.ops.aten.mul.Tensor(mul_303, unsqueeze_701);  mul_303 = unsqueeze_701 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_189 = torch.ops.aten.add.Tensor(mul_304, unsqueeze_703);  mul_304 = unsqueeze_703 = None
        sigmoid_41 = torch.ops.aten.sigmoid.default(add_189)
        mul_305 = torch.ops.aten.mul.Tensor(add_189, sigmoid_41);  add_189 = sigmoid_41 = None
        convolution_105 = torch.ops.aten.convolution.default(mul_305, arg143_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 504);  mul_305 = arg143_1 = None
        add_190 = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_88 = torch.ops.aten.sqrt.default(add_190);  add_190 = None
        reciprocal_88 = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_306 = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88 = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_705);  convolution_105 = unsqueeze_705 = None
        mul_307 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_709);  mul_307 = unsqueeze_709 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_191 = torch.ops.aten.add.Tensor(mul_308, unsqueeze_711);  mul_308 = unsqueeze_711 = None
        mean_18 = torch.ops.aten.mean.dim(add_191, [2, 3], True)
        convolution_106 = torch.ops.aten.convolution.default(mean_18, arg148_1, arg149_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg148_1 = arg149_1 = None
        add_192 = torch.ops.aten.add.Tensor(arg151_1, 1e-05);  arg151_1 = None
        sqrt_89 = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_89 = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_309 = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89 = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_713);  convolution_106 = unsqueeze_713 = None
        mul_310 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_717);  mul_310 = unsqueeze_717 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_193 = torch.ops.aten.add.Tensor(mul_311, unsqueeze_719);  mul_311 = unsqueeze_719 = None
        relu_17 = torch.ops.aten.relu.default(add_193);  add_193 = None
        convolution_107 = torch.ops.aten.convolution.default(relu_17, arg154_1, arg155_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg154_1 = arg155_1 = None
        sigmoid_42 = torch.ops.aten.sigmoid.default(convolution_107);  convolution_107 = None
        mul_312 = torch.ops.aten.mul.Tensor(add_191, sigmoid_42);  add_191 = sigmoid_42 = None
        clamp_min_23 = torch.ops.aten.clamp_min.default(mul_312, 0.0);  mul_312 = None
        clamp_max_23 = torch.ops.aten.clamp_max.default(clamp_min_23, 6.0);  clamp_min_23 = None
        convolution_108 = torch.ops.aten.convolution.default(clamp_max_23, arg156_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_23 = arg156_1 = None
        add_194 = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_90 = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_90 = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_313 = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(mul_313, -1);  mul_313 = None
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90 = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_721);  convolution_108 = unsqueeze_721 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_725);  mul_314 = unsqueeze_725 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_195 = torch.ops.aten.add.Tensor(mul_315, unsqueeze_727);  mul_315 = unsqueeze_727 = None
        slice_58 = torch.ops.aten.slice.Tensor(add_195, 1, 0, 84)
        add_196 = torch.ops.aten.add.Tensor(slice_58, cat_13);  slice_58 = cat_13 = None
        slice_60 = torch.ops.aten.slice.Tensor(add_195, 1, 84, 9223372036854775807);  add_195 = None
        cat_14 = torch.ops.aten.cat.default([add_196, slice_60], 1);  add_196 = slice_60 = None
        convolution_109 = torch.ops.aten.convolution.default(cat_14, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg161_1 = None
        add_197 = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_91 = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_91 = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_316 = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(mul_316, -1);  mul_316 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_729);  convolution_109 = unsqueeze_729 = None
        mul_317 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_318 = torch.ops.aten.mul.Tensor(mul_317, unsqueeze_733);  mul_317 = unsqueeze_733 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_198 = torch.ops.aten.add.Tensor(mul_318, unsqueeze_735);  mul_318 = unsqueeze_735 = None
        sigmoid_43 = torch.ops.aten.sigmoid.default(add_198)
        mul_319 = torch.ops.aten.mul.Tensor(add_198, sigmoid_43);  add_198 = sigmoid_43 = None
        convolution_110 = torch.ops.aten.convolution.default(mul_319, arg166_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 570);  mul_319 = arg166_1 = None
        add_199 = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_92 = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_92 = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_320 = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_320, -1);  mul_320 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92 = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_737);  convolution_110 = unsqueeze_737 = None
        mul_321 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_322 = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_741);  mul_321 = unsqueeze_741 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_200 = torch.ops.aten.add.Tensor(mul_322, unsqueeze_743);  mul_322 = unsqueeze_743 = None
        mean_19 = torch.ops.aten.mean.dim(add_200, [2, 3], True)
        convolution_111 = torch.ops.aten.convolution.default(mean_19, arg171_1, arg172_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg171_1 = arg172_1 = None
        add_201 = torch.ops.aten.add.Tensor(arg174_1, 1e-05);  arg174_1 = None
        sqrt_93 = torch.ops.aten.sqrt.default(add_201);  add_201 = None
        reciprocal_93 = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_323 = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(mul_323, -1);  mul_323 = None
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_745);  convolution_111 = unsqueeze_745 = None
        mul_324 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_325 = torch.ops.aten.mul.Tensor(mul_324, unsqueeze_749);  mul_324 = unsqueeze_749 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(arg176_1, -1);  arg176_1 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_202 = torch.ops.aten.add.Tensor(mul_325, unsqueeze_751);  mul_325 = unsqueeze_751 = None
        relu_18 = torch.ops.aten.relu.default(add_202);  add_202 = None
        convolution_112 = torch.ops.aten.convolution.default(relu_18, arg177_1, arg178_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_18 = arg177_1 = arg178_1 = None
        sigmoid_44 = torch.ops.aten.sigmoid.default(convolution_112);  convolution_112 = None
        mul_326 = torch.ops.aten.mul.Tensor(add_200, sigmoid_44);  add_200 = sigmoid_44 = None
        clamp_min_24 = torch.ops.aten.clamp_min.default(mul_326, 0.0);  mul_326 = None
        clamp_max_24 = torch.ops.aten.clamp_max.default(clamp_min_24, 6.0);  clamp_min_24 = None
        convolution_113 = torch.ops.aten.convolution.default(clamp_max_24, arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_24 = arg179_1 = None
        add_203 = torch.ops.aten.add.Tensor(arg181_1, 1e-05);  arg181_1 = None
        sqrt_94 = torch.ops.aten.sqrt.default(add_203);  add_203 = None
        reciprocal_94 = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_327 = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94 = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_753);  convolution_113 = unsqueeze_753 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_757);  mul_328 = unsqueeze_757 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_204 = torch.ops.aten.add.Tensor(mul_329, unsqueeze_759);  mul_329 = unsqueeze_759 = None
        slice_62 = torch.ops.aten.slice.Tensor(add_204, 1, 0, 95)
        add_205 = torch.ops.aten.add.Tensor(slice_62, cat_14);  slice_62 = cat_14 = None
        slice_64 = torch.ops.aten.slice.Tensor(add_204, 1, 95, 9223372036854775807);  add_204 = None
        cat_15 = torch.ops.aten.cat.default([add_205, slice_64], 1);  add_205 = slice_64 = None
        convolution_114 = torch.ops.aten.convolution.default(cat_15, arg184_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg184_1 = None
        add_206 = torch.ops.aten.add.Tensor(arg186_1, 1e-05);  arg186_1 = None
        sqrt_95 = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_95 = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_330 = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95 = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_761);  convolution_114 = unsqueeze_761 = None
        mul_331 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_765);  mul_331 = unsqueeze_765 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(arg188_1, -1);  arg188_1 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_207 = torch.ops.aten.add.Tensor(mul_332, unsqueeze_767);  mul_332 = unsqueeze_767 = None
        sigmoid_45 = torch.ops.aten.sigmoid.default(add_207)
        mul_333 = torch.ops.aten.mul.Tensor(add_207, sigmoid_45);  add_207 = sigmoid_45 = None
        convolution_115 = torch.ops.aten.convolution.default(mul_333, arg189_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 636);  mul_333 = arg189_1 = None
        add_208 = torch.ops.aten.add.Tensor(arg191_1, 1e-05);  arg191_1 = None
        sqrt_96 = torch.ops.aten.sqrt.default(add_208);  add_208 = None
        reciprocal_96 = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_334 = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(mul_334, -1);  mul_334 = None
        unsqueeze_771 = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96 = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_769);  convolution_115 = unsqueeze_769 = None
        mul_335 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_773 = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_336 = torch.ops.aten.mul.Tensor(mul_335, unsqueeze_773);  mul_335 = unsqueeze_773 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(arg193_1, -1);  arg193_1 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_209 = torch.ops.aten.add.Tensor(mul_336, unsqueeze_775);  mul_336 = unsqueeze_775 = None
        mean_20 = torch.ops.aten.mean.dim(add_209, [2, 3], True)
        convolution_116 = torch.ops.aten.convolution.default(mean_20, arg194_1, arg195_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg194_1 = arg195_1 = None
        add_210 = torch.ops.aten.add.Tensor(arg197_1, 1e-05);  arg197_1 = None
        sqrt_97 = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_97 = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_337 = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(arg196_1, -1);  arg196_1 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(mul_337, -1);  mul_337 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97 = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_777);  convolution_116 = unsqueeze_777 = None
        mul_338 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_339 = torch.ops.aten.mul.Tensor(mul_338, unsqueeze_781);  mul_338 = unsqueeze_781 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_211 = torch.ops.aten.add.Tensor(mul_339, unsqueeze_783);  mul_339 = unsqueeze_783 = None
        relu_19 = torch.ops.aten.relu.default(add_211);  add_211 = None
        convolution_117 = torch.ops.aten.convolution.default(relu_19, arg200_1, arg201_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_19 = arg200_1 = arg201_1 = None
        sigmoid_46 = torch.ops.aten.sigmoid.default(convolution_117);  convolution_117 = None
        mul_340 = torch.ops.aten.mul.Tensor(add_209, sigmoid_46);  add_209 = sigmoid_46 = None
        clamp_min_25 = torch.ops.aten.clamp_min.default(mul_340, 0.0);  mul_340 = None
        clamp_max_25 = torch.ops.aten.clamp_max.default(clamp_min_25, 6.0);  clamp_min_25 = None
        convolution_118 = torch.ops.aten.convolution.default(clamp_max_25, arg202_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_25 = arg202_1 = None
        add_212 = torch.ops.aten.add.Tensor(arg204_1, 1e-05);  arg204_1 = None
        sqrt_98 = torch.ops.aten.sqrt.default(add_212);  add_212 = None
        reciprocal_98 = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_341 = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_341, -1);  mul_341 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98 = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_785);  convolution_118 = unsqueeze_785 = None
        mul_342 = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_343 = torch.ops.aten.mul.Tensor(mul_342, unsqueeze_789);  mul_342 = unsqueeze_789 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(arg206_1, -1);  arg206_1 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_213 = torch.ops.aten.add.Tensor(mul_343, unsqueeze_791);  mul_343 = unsqueeze_791 = None
        slice_66 = torch.ops.aten.slice.Tensor(add_213, 1, 0, 106)
        add_214 = torch.ops.aten.add.Tensor(slice_66, cat_15);  slice_66 = cat_15 = None
        slice_68 = torch.ops.aten.slice.Tensor(add_213, 1, 106, 9223372036854775807);  add_213 = None
        cat_16 = torch.ops.aten.cat.default([add_214, slice_68], 1);  add_214 = slice_68 = None
        convolution_119 = torch.ops.aten.convolution.default(cat_16, arg207_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg207_1 = None
        add_215 = torch.ops.aten.add.Tensor(arg209_1, 1e-05);  arg209_1 = None
        sqrt_99 = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_99 = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_344 = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(mul_344, -1);  mul_344 = None
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99 = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_793);  convolution_119 = unsqueeze_793 = None
        mul_345 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_346 = torch.ops.aten.mul.Tensor(mul_345, unsqueeze_797);  mul_345 = unsqueeze_797 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_216 = torch.ops.aten.add.Tensor(mul_346, unsqueeze_799);  mul_346 = unsqueeze_799 = None
        sigmoid_47 = torch.ops.aten.sigmoid.default(add_216)
        mul_347 = torch.ops.aten.mul.Tensor(add_216, sigmoid_47);  add_216 = sigmoid_47 = None
        convolution_120 = torch.ops.aten.convolution.default(mul_347, arg212_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 702);  mul_347 = arg212_1 = None
        add_217 = torch.ops.aten.add.Tensor(arg214_1, 1e-05);  arg214_1 = None
        sqrt_100 = torch.ops.aten.sqrt.default(add_217);  add_217 = None
        reciprocal_100 = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_348 = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100 = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_801);  convolution_120 = unsqueeze_801 = None
        mul_349 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_805);  mul_349 = unsqueeze_805 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_218 = torch.ops.aten.add.Tensor(mul_350, unsqueeze_807);  mul_350 = unsqueeze_807 = None
        mean_21 = torch.ops.aten.mean.dim(add_218, [2, 3], True)
        convolution_121 = torch.ops.aten.convolution.default(mean_21, arg217_1, arg218_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg217_1 = arg218_1 = None
        add_219 = torch.ops.aten.add.Tensor(arg220_1, 1e-05);  arg220_1 = None
        sqrt_101 = torch.ops.aten.sqrt.default(add_219);  add_219 = None
        reciprocal_101 = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_351 = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101 = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_809);  convolution_121 = unsqueeze_809 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_813);  mul_352 = unsqueeze_813 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_220 = torch.ops.aten.add.Tensor(mul_353, unsqueeze_815);  mul_353 = unsqueeze_815 = None
        relu_20 = torch.ops.aten.relu.default(add_220);  add_220 = None
        convolution_122 = torch.ops.aten.convolution.default(relu_20, arg223_1, arg224_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = arg223_1 = arg224_1 = None
        sigmoid_48 = torch.ops.aten.sigmoid.default(convolution_122);  convolution_122 = None
        mul_354 = torch.ops.aten.mul.Tensor(add_218, sigmoid_48);  add_218 = sigmoid_48 = None
        clamp_min_26 = torch.ops.aten.clamp_min.default(mul_354, 0.0);  mul_354 = None
        clamp_max_26 = torch.ops.aten.clamp_max.default(clamp_min_26, 6.0);  clamp_min_26 = None
        convolution_123 = torch.ops.aten.convolution.default(clamp_max_26, arg225_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_26 = arg225_1 = None
        add_221 = torch.ops.aten.add.Tensor(arg227_1, 1e-05);  arg227_1 = None
        sqrt_102 = torch.ops.aten.sqrt.default(add_221);  add_221 = None
        reciprocal_102 = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_355 = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(arg226_1, -1);  arg226_1 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(mul_355, -1);  mul_355 = None
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102 = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_817);  convolution_123 = unsqueeze_817 = None
        mul_356 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(arg228_1, -1);  arg228_1 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_357 = torch.ops.aten.mul.Tensor(mul_356, unsqueeze_821);  mul_356 = unsqueeze_821 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_222 = torch.ops.aten.add.Tensor(mul_357, unsqueeze_823);  mul_357 = unsqueeze_823 = None
        slice_70 = torch.ops.aten.slice.Tensor(add_222, 1, 0, 117)
        add_223 = torch.ops.aten.add.Tensor(slice_70, cat_16);  slice_70 = cat_16 = None
        slice_72 = torch.ops.aten.slice.Tensor(add_222, 1, 117, 9223372036854775807);  add_222 = None
        cat_17 = torch.ops.aten.cat.default([add_223, slice_72], 1);  add_223 = slice_72 = None
        convolution_124 = torch.ops.aten.convolution.default(cat_17, arg230_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_17 = arg230_1 = None
        add_224 = torch.ops.aten.add.Tensor(arg232_1, 1e-05);  arg232_1 = None
        sqrt_103 = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_103 = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_358 = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(arg231_1, -1);  arg231_1 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(mul_358, -1);  mul_358 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103 = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_825);  convolution_124 = unsqueeze_825 = None
        mul_359 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(arg233_1, -1);  arg233_1 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_360 = torch.ops.aten.mul.Tensor(mul_359, unsqueeze_829);  mul_359 = unsqueeze_829 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_225 = torch.ops.aten.add.Tensor(mul_360, unsqueeze_831);  mul_360 = unsqueeze_831 = None
        sigmoid_49 = torch.ops.aten.sigmoid.default(add_225)
        mul_361 = torch.ops.aten.mul.Tensor(add_225, sigmoid_49);  add_225 = sigmoid_49 = None
        convolution_125 = torch.ops.aten.convolution.default(mul_361, arg235_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 768);  mul_361 = arg235_1 = None
        add_226 = torch.ops.aten.add.Tensor(arg237_1, 1e-05);  arg237_1 = None
        sqrt_104 = torch.ops.aten.sqrt.default(add_226);  add_226 = None
        reciprocal_104 = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_362 = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832 = torch.ops.aten.unsqueeze.default(arg236_1, -1);  arg236_1 = None
        unsqueeze_833 = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(mul_362, -1);  mul_362 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104 = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_833);  convolution_125 = unsqueeze_833 = None
        mul_363 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(arg238_1, -1);  arg238_1 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_364 = torch.ops.aten.mul.Tensor(mul_363, unsqueeze_837);  mul_363 = unsqueeze_837 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_227 = torch.ops.aten.add.Tensor(mul_364, unsqueeze_839);  mul_364 = unsqueeze_839 = None
        mean_22 = torch.ops.aten.mean.dim(add_227, [2, 3], True)
        convolution_126 = torch.ops.aten.convolution.default(mean_22, arg240_1, arg241_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg240_1 = arg241_1 = None
        add_228 = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_105 = torch.ops.aten.sqrt.default(add_228);  add_228 = None
        reciprocal_105 = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_365 = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(mul_365, -1);  mul_365 = None
        unsqueeze_843 = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105 = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_841);  convolution_126 = unsqueeze_841 = None
        mul_366 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_845 = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_367 = torch.ops.aten.mul.Tensor(mul_366, unsqueeze_845);  mul_366 = unsqueeze_845 = None
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_229 = torch.ops.aten.add.Tensor(mul_367, unsqueeze_847);  mul_367 = unsqueeze_847 = None
        relu_21 = torch.ops.aten.relu.default(add_229);  add_229 = None
        convolution_127 = torch.ops.aten.convolution.default(relu_21, arg246_1, arg247_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = arg246_1 = arg247_1 = None
        sigmoid_50 = torch.ops.aten.sigmoid.default(convolution_127);  convolution_127 = None
        mul_368 = torch.ops.aten.mul.Tensor(add_227, sigmoid_50);  add_227 = sigmoid_50 = None
        clamp_min_27 = torch.ops.aten.clamp_min.default(mul_368, 0.0);  mul_368 = None
        clamp_max_27 = torch.ops.aten.clamp_max.default(clamp_min_27, 6.0);  clamp_min_27 = None
        convolution_128 = torch.ops.aten.convolution.default(clamp_max_27, arg248_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_27 = arg248_1 = None
        add_230 = torch.ops.aten.add.Tensor(arg250_1, 1e-05);  arg250_1 = None
        sqrt_106 = torch.ops.aten.sqrt.default(add_230);  add_230 = None
        reciprocal_106 = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_369 = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106 = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_849);  convolution_128 = unsqueeze_849 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(arg251_1, -1);  arg251_1 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_853);  mul_370 = unsqueeze_853 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_855 = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_231 = torch.ops.aten.add.Tensor(mul_371, unsqueeze_855);  mul_371 = unsqueeze_855 = None
        convolution_129 = torch.ops.aten.convolution.default(add_231, arg253_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg253_1 = None
        add_232 = torch.ops.aten.add.Tensor(arg255_1, 1e-05);  arg255_1 = None
        sqrt_107 = torch.ops.aten.sqrt.default(add_232);  add_232 = None
        reciprocal_107 = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_372 = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_857 = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107 = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_857);  convolution_129 = unsqueeze_857 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(arg256_1, -1);  arg256_1 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_861);  mul_373 = unsqueeze_861 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_233 = torch.ops.aten.add.Tensor(mul_374, unsqueeze_863);  mul_374 = unsqueeze_863 = None
        sigmoid_51 = torch.ops.aten.sigmoid.default(add_233)
        mul_375 = torch.ops.aten.mul.Tensor(add_233, sigmoid_51);  add_233 = sigmoid_51 = None
        convolution_130 = torch.ops.aten.convolution.default(mul_375, arg258_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 840);  mul_375 = arg258_1 = None
        add_234 = torch.ops.aten.add.Tensor(arg260_1, 1e-05);  arg260_1 = None
        sqrt_108 = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_108 = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_376 = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864 = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_865 = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866 = torch.ops.aten.unsqueeze.default(mul_376, -1);  mul_376 = None
        unsqueeze_867 = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108 = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_865);  convolution_130 = unsqueeze_865 = None
        mul_377 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868 = torch.ops.aten.unsqueeze.default(arg261_1, -1);  arg261_1 = None
        unsqueeze_869 = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_378 = torch.ops.aten.mul.Tensor(mul_377, unsqueeze_869);  mul_377 = unsqueeze_869 = None
        unsqueeze_870 = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_871 = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_235 = torch.ops.aten.add.Tensor(mul_378, unsqueeze_871);  mul_378 = unsqueeze_871 = None
        mean_23 = torch.ops.aten.mean.dim(add_235, [2, 3], True)
        convolution_131 = torch.ops.aten.convolution.default(mean_23, arg263_1, arg264_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg263_1 = arg264_1 = None
        add_236 = torch.ops.aten.add.Tensor(arg266_1, 1e-05);  arg266_1 = None
        sqrt_109 = torch.ops.aten.sqrt.default(add_236);  add_236 = None
        reciprocal_109 = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_379 = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_873 = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874 = torch.ops.aten.unsqueeze.default(mul_379, -1);  mul_379 = None
        unsqueeze_875 = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109 = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_873);  convolution_131 = unsqueeze_873 = None
        mul_380 = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876 = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_877 = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_381 = torch.ops.aten.mul.Tensor(mul_380, unsqueeze_877);  mul_380 = unsqueeze_877 = None
        unsqueeze_878 = torch.ops.aten.unsqueeze.default(arg268_1, -1);  arg268_1 = None
        unsqueeze_879 = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_237 = torch.ops.aten.add.Tensor(mul_381, unsqueeze_879);  mul_381 = unsqueeze_879 = None
        relu_22 = torch.ops.aten.relu.default(add_237);  add_237 = None
        convolution_132 = torch.ops.aten.convolution.default(relu_22, arg269_1, arg270_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_22 = arg269_1 = arg270_1 = None
        sigmoid_52 = torch.ops.aten.sigmoid.default(convolution_132);  convolution_132 = None
        mul_382 = torch.ops.aten.mul.Tensor(add_235, sigmoid_52);  add_235 = sigmoid_52 = None
        clamp_min_28 = torch.ops.aten.clamp_min.default(mul_382, 0.0);  mul_382 = None
        clamp_max_28 = torch.ops.aten.clamp_max.default(clamp_min_28, 6.0);  clamp_min_28 = None
        convolution_133 = torch.ops.aten.convolution.default(clamp_max_28, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_28 = arg271_1 = None
        add_238 = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_110 = torch.ops.aten.sqrt.default(add_238);  add_238 = None
        reciprocal_110 = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_383 = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880 = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_881 = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882 = torch.ops.aten.unsqueeze.default(mul_383, -1);  mul_383 = None
        unsqueeze_883 = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110 = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_881);  convolution_133 = unsqueeze_881 = None
        mul_384 = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884 = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_885 = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_385 = torch.ops.aten.mul.Tensor(mul_384, unsqueeze_885);  mul_384 = unsqueeze_885 = None
        unsqueeze_886 = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_887 = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_239 = torch.ops.aten.add.Tensor(mul_385, unsqueeze_887);  mul_385 = unsqueeze_887 = None
        slice_74 = torch.ops.aten.slice.Tensor(add_239, 1, 0, 140)
        add_240 = torch.ops.aten.add.Tensor(slice_74, add_231);  slice_74 = add_231 = None
        slice_76 = torch.ops.aten.slice.Tensor(add_239, 1, 140, 9223372036854775807);  add_239 = None
        cat_18 = torch.ops.aten.cat.default([add_240, slice_76], 1);  add_240 = slice_76 = None
        convolution_134 = torch.ops.aten.convolution.default(cat_18, arg276_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg276_1 = None
        add_241 = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
        sqrt_111 = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_111 = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_386 = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(mul_386, -1);  mul_386 = None
        unsqueeze_891 = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111 = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_889);  convolution_134 = unsqueeze_889 = None
        mul_387 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892 = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_893 = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_388 = torch.ops.aten.mul.Tensor(mul_387, unsqueeze_893);  mul_387 = unsqueeze_893 = None
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_242 = torch.ops.aten.add.Tensor(mul_388, unsqueeze_895);  mul_388 = unsqueeze_895 = None
        sigmoid_53 = torch.ops.aten.sigmoid.default(add_242)
        mul_389 = torch.ops.aten.mul.Tensor(add_242, sigmoid_53);  add_242 = sigmoid_53 = None
        convolution_135 = torch.ops.aten.convolution.default(mul_389, arg281_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 906);  mul_389 = arg281_1 = None
        add_243 = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_112 = torch.ops.aten.sqrt.default(add_243);  add_243 = None
        reciprocal_112 = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_390 = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112 = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_897);  convolution_135 = unsqueeze_897 = None
        mul_391 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_392 = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_901);  mul_391 = unsqueeze_901 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_903 = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_244 = torch.ops.aten.add.Tensor(mul_392, unsqueeze_903);  mul_392 = unsqueeze_903 = None
        mean_24 = torch.ops.aten.mean.dim(add_244, [2, 3], True)
        convolution_136 = torch.ops.aten.convolution.default(mean_24, arg286_1, arg287_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg286_1 = arg287_1 = None
        add_245 = torch.ops.aten.add.Tensor(arg289_1, 1e-05);  arg289_1 = None
        sqrt_113 = torch.ops.aten.sqrt.default(add_245);  add_245 = None
        reciprocal_113 = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_393 = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904 = torch.ops.aten.unsqueeze.default(arg288_1, -1);  arg288_1 = None
        unsqueeze_905 = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113 = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_905);  convolution_136 = unsqueeze_905 = None
        mul_394 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_395 = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_909);  mul_394 = unsqueeze_909 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(arg291_1, -1);  arg291_1 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_246 = torch.ops.aten.add.Tensor(mul_395, unsqueeze_911);  mul_395 = unsqueeze_911 = None
        relu_23 = torch.ops.aten.relu.default(add_246);  add_246 = None
        convolution_137 = torch.ops.aten.convolution.default(relu_23, arg292_1, arg293_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg292_1 = arg293_1 = None
        sigmoid_54 = torch.ops.aten.sigmoid.default(convolution_137);  convolution_137 = None
        mul_396 = torch.ops.aten.mul.Tensor(add_244, sigmoid_54);  add_244 = sigmoid_54 = None
        clamp_min_29 = torch.ops.aten.clamp_min.default(mul_396, 0.0);  mul_396 = None
        clamp_max_29 = torch.ops.aten.clamp_max.default(clamp_min_29, 6.0);  clamp_min_29 = None
        convolution_138 = torch.ops.aten.convolution.default(clamp_max_29, arg294_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_29 = arg294_1 = None
        add_247 = torch.ops.aten.add.Tensor(arg296_1, 1e-05);  arg296_1 = None
        sqrt_114 = torch.ops.aten.sqrt.default(add_247);  add_247 = None
        reciprocal_114 = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_397 = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912 = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_913 = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914 = torch.ops.aten.unsqueeze.default(mul_397, -1);  mul_397 = None
        unsqueeze_915 = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114 = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_913);  convolution_138 = unsqueeze_913 = None
        mul_398 = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916 = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_917 = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_399 = torch.ops.aten.mul.Tensor(mul_398, unsqueeze_917);  mul_398 = unsqueeze_917 = None
        unsqueeze_918 = torch.ops.aten.unsqueeze.default(arg298_1, -1);  arg298_1 = None
        unsqueeze_919 = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_248 = torch.ops.aten.add.Tensor(mul_399, unsqueeze_919);  mul_399 = unsqueeze_919 = None
        slice_78 = torch.ops.aten.slice.Tensor(add_248, 1, 0, 151)
        add_249 = torch.ops.aten.add.Tensor(slice_78, cat_18);  slice_78 = cat_18 = None
        slice_80 = torch.ops.aten.slice.Tensor(add_248, 1, 151, 9223372036854775807);  add_248 = None
        cat_19 = torch.ops.aten.cat.default([add_249, slice_80], 1);  add_249 = slice_80 = None
        convolution_139 = torch.ops.aten.convolution.default(cat_19, arg299_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg299_1 = None
        add_250 = torch.ops.aten.add.Tensor(arg301_1, 1e-05);  arg301_1 = None
        sqrt_115 = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_115 = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_400 = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920 = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_921 = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922 = torch.ops.aten.unsqueeze.default(mul_400, -1);  mul_400 = None
        unsqueeze_923 = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115 = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_921);  convolution_139 = unsqueeze_921 = None
        mul_401 = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924 = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_925 = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_402 = torch.ops.aten.mul.Tensor(mul_401, unsqueeze_925);  mul_401 = unsqueeze_925 = None
        unsqueeze_926 = torch.ops.aten.unsqueeze.default(arg303_1, -1);  arg303_1 = None
        unsqueeze_927 = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_251 = torch.ops.aten.add.Tensor(mul_402, unsqueeze_927);  mul_402 = unsqueeze_927 = None
        sigmoid_55 = torch.ops.aten.sigmoid.default(add_251)
        mul_403 = torch.ops.aten.mul.Tensor(add_251, sigmoid_55);  add_251 = sigmoid_55 = None
        convolution_140 = torch.ops.aten.convolution.default(mul_403, arg304_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 972);  mul_403 = arg304_1 = None
        add_252 = torch.ops.aten.add.Tensor(arg306_1, 1e-05);  arg306_1 = None
        sqrt_116 = torch.ops.aten.sqrt.default(add_252);  add_252 = None
        reciprocal_116 = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_404 = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_928 = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_929 = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        unsqueeze_930 = torch.ops.aten.unsqueeze.default(mul_404, -1);  mul_404 = None
        unsqueeze_931 = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        sub_116 = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_929);  convolution_140 = unsqueeze_929 = None
        mul_405 = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932 = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
        unsqueeze_933 = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_406 = torch.ops.aten.mul.Tensor(mul_405, unsqueeze_933);  mul_405 = unsqueeze_933 = None
        unsqueeze_934 = torch.ops.aten.unsqueeze.default(arg308_1, -1);  arg308_1 = None
        unsqueeze_935 = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_253 = torch.ops.aten.add.Tensor(mul_406, unsqueeze_935);  mul_406 = unsqueeze_935 = None
        mean_25 = torch.ops.aten.mean.dim(add_253, [2, 3], True)
        convolution_141 = torch.ops.aten.convolution.default(mean_25, arg309_1, arg310_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_25 = arg309_1 = arg310_1 = None
        add_254 = torch.ops.aten.add.Tensor(arg312_1, 1e-05);  arg312_1 = None
        sqrt_117 = torch.ops.aten.sqrt.default(add_254);  add_254 = None
        reciprocal_117 = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_407 = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_936 = torch.ops.aten.unsqueeze.default(arg311_1, -1);  arg311_1 = None
        unsqueeze_937 = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        unsqueeze_938 = torch.ops.aten.unsqueeze.default(mul_407, -1);  mul_407 = None
        unsqueeze_939 = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        sub_117 = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_937);  convolution_141 = unsqueeze_937 = None
        mul_408 = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940 = torch.ops.aten.unsqueeze.default(arg313_1, -1);  arg313_1 = None
        unsqueeze_941 = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_409 = torch.ops.aten.mul.Tensor(mul_408, unsqueeze_941);  mul_408 = unsqueeze_941 = None
        unsqueeze_942 = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_943 = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_255 = torch.ops.aten.add.Tensor(mul_409, unsqueeze_943);  mul_409 = unsqueeze_943 = None
        relu_24 = torch.ops.aten.relu.default(add_255);  add_255 = None
        convolution_142 = torch.ops.aten.convolution.default(relu_24, arg315_1, arg316_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_24 = arg315_1 = arg316_1 = None
        sigmoid_56 = torch.ops.aten.sigmoid.default(convolution_142);  convolution_142 = None
        mul_410 = torch.ops.aten.mul.Tensor(add_253, sigmoid_56);  add_253 = sigmoid_56 = None
        clamp_min_30 = torch.ops.aten.clamp_min.default(mul_410, 0.0);  mul_410 = None
        clamp_max_30 = torch.ops.aten.clamp_max.default(clamp_min_30, 6.0);  clamp_min_30 = None
        convolution_143 = torch.ops.aten.convolution.default(clamp_max_30, arg317_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_30 = arg317_1 = None
        add_256 = torch.ops.aten.add.Tensor(arg319_1, 1e-05);  arg319_1 = None
        sqrt_118 = torch.ops.aten.sqrt.default(add_256);  add_256 = None
        reciprocal_118 = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_411 = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_944 = torch.ops.aten.unsqueeze.default(arg318_1, -1);  arg318_1 = None
        unsqueeze_945 = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        unsqueeze_946 = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_947 = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        sub_118 = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_945);  convolution_143 = unsqueeze_945 = None
        mul_412 = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948 = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_949 = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_413 = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_949);  mul_412 = unsqueeze_949 = None
        unsqueeze_950 = torch.ops.aten.unsqueeze.default(arg321_1, -1);  arg321_1 = None
        unsqueeze_951 = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_257 = torch.ops.aten.add.Tensor(mul_413, unsqueeze_951);  mul_413 = unsqueeze_951 = None
        slice_82 = torch.ops.aten.slice.Tensor(add_257, 1, 0, 162)
        add_258 = torch.ops.aten.add.Tensor(slice_82, cat_19);  slice_82 = cat_19 = None
        slice_84 = torch.ops.aten.slice.Tensor(add_257, 1, 162, 9223372036854775807);  add_257 = None
        cat_20 = torch.ops.aten.cat.default([add_258, slice_84], 1);  add_258 = slice_84 = None
        convolution_144 = torch.ops.aten.convolution.default(cat_20, arg322_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg322_1 = None
        add_259 = torch.ops.aten.add.Tensor(arg324_1, 1e-05);  arg324_1 = None
        sqrt_119 = torch.ops.aten.sqrt.default(add_259);  add_259 = None
        reciprocal_119 = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_414 = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_952 = torch.ops.aten.unsqueeze.default(arg323_1, -1);  arg323_1 = None
        unsqueeze_953 = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        unsqueeze_954 = torch.ops.aten.unsqueeze.default(mul_414, -1);  mul_414 = None
        unsqueeze_955 = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        sub_119 = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_953);  convolution_144 = unsqueeze_953 = None
        mul_415 = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956 = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_957 = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_416 = torch.ops.aten.mul.Tensor(mul_415, unsqueeze_957);  mul_415 = unsqueeze_957 = None
        unsqueeze_958 = torch.ops.aten.unsqueeze.default(arg326_1, -1);  arg326_1 = None
        unsqueeze_959 = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_260 = torch.ops.aten.add.Tensor(mul_416, unsqueeze_959);  mul_416 = unsqueeze_959 = None
        sigmoid_57 = torch.ops.aten.sigmoid.default(add_260)
        mul_417 = torch.ops.aten.mul.Tensor(add_260, sigmoid_57);  add_260 = sigmoid_57 = None
        convolution_145 = torch.ops.aten.convolution.default(mul_417, arg327_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1044);  mul_417 = arg327_1 = None
        add_261 = torch.ops.aten.add.Tensor(arg329_1, 1e-05);  arg329_1 = None
        sqrt_120 = torch.ops.aten.sqrt.default(add_261);  add_261 = None
        reciprocal_120 = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_418 = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_960 = torch.ops.aten.unsqueeze.default(arg328_1, -1);  arg328_1 = None
        unsqueeze_961 = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        unsqueeze_962 = torch.ops.aten.unsqueeze.default(mul_418, -1);  mul_418 = None
        unsqueeze_963 = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        sub_120 = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_961);  convolution_145 = unsqueeze_961 = None
        mul_419 = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964 = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_965 = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_420 = torch.ops.aten.mul.Tensor(mul_419, unsqueeze_965);  mul_419 = unsqueeze_965 = None
        unsqueeze_966 = torch.ops.aten.unsqueeze.default(arg331_1, -1);  arg331_1 = None
        unsqueeze_967 = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_262 = torch.ops.aten.add.Tensor(mul_420, unsqueeze_967);  mul_420 = unsqueeze_967 = None
        mean_26 = torch.ops.aten.mean.dim(add_262, [2, 3], True)
        convolution_146 = torch.ops.aten.convolution.default(mean_26, arg332_1, arg333_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_26 = arg332_1 = arg333_1 = None
        add_263 = torch.ops.aten.add.Tensor(arg335_1, 1e-05);  arg335_1 = None
        sqrt_121 = torch.ops.aten.sqrt.default(add_263);  add_263 = None
        reciprocal_121 = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_421 = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_968 = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_969 = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        unsqueeze_970 = torch.ops.aten.unsqueeze.default(mul_421, -1);  mul_421 = None
        unsqueeze_971 = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        sub_121 = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_969);  convolution_146 = unsqueeze_969 = None
        mul_422 = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972 = torch.ops.aten.unsqueeze.default(arg336_1, -1);  arg336_1 = None
        unsqueeze_973 = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_423 = torch.ops.aten.mul.Tensor(mul_422, unsqueeze_973);  mul_422 = unsqueeze_973 = None
        unsqueeze_974 = torch.ops.aten.unsqueeze.default(arg337_1, -1);  arg337_1 = None
        unsqueeze_975 = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_264 = torch.ops.aten.add.Tensor(mul_423, unsqueeze_975);  mul_423 = unsqueeze_975 = None
        relu_25 = torch.ops.aten.relu.default(add_264);  add_264 = None
        convolution_147 = torch.ops.aten.convolution.default(relu_25, arg338_1, arg339_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_25 = arg338_1 = arg339_1 = None
        sigmoid_58 = torch.ops.aten.sigmoid.default(convolution_147);  convolution_147 = None
        mul_424 = torch.ops.aten.mul.Tensor(add_262, sigmoid_58);  add_262 = sigmoid_58 = None
        clamp_min_31 = torch.ops.aten.clamp_min.default(mul_424, 0.0);  mul_424 = None
        clamp_max_31 = torch.ops.aten.clamp_max.default(clamp_min_31, 6.0);  clamp_min_31 = None
        convolution_148 = torch.ops.aten.convolution.default(clamp_max_31, arg340_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_31 = arg340_1 = None
        add_265 = torch.ops.aten.add.Tensor(arg342_1, 1e-05);  arg342_1 = None
        sqrt_122 = torch.ops.aten.sqrt.default(add_265);  add_265 = None
        reciprocal_122 = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_425 = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_976 = torch.ops.aten.unsqueeze.default(arg341_1, -1);  arg341_1 = None
        unsqueeze_977 = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        unsqueeze_978 = torch.ops.aten.unsqueeze.default(mul_425, -1);  mul_425 = None
        unsqueeze_979 = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        sub_122 = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_977);  convolution_148 = unsqueeze_977 = None
        mul_426 = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980 = torch.ops.aten.unsqueeze.default(arg343_1, -1);  arg343_1 = None
        unsqueeze_981 = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_427 = torch.ops.aten.mul.Tensor(mul_426, unsqueeze_981);  mul_426 = unsqueeze_981 = None
        unsqueeze_982 = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_983 = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_266 = torch.ops.aten.add.Tensor(mul_427, unsqueeze_983);  mul_427 = unsqueeze_983 = None
        slice_86 = torch.ops.aten.slice.Tensor(add_266, 1, 0, 174)
        add_267 = torch.ops.aten.add.Tensor(slice_86, cat_20);  slice_86 = cat_20 = None
        slice_88 = torch.ops.aten.slice.Tensor(add_266, 1, 174, 9223372036854775807);  add_266 = None
        cat_21 = torch.ops.aten.cat.default([add_267, slice_88], 1);  add_267 = slice_88 = None
        convolution_149 = torch.ops.aten.convolution.default(cat_21, arg345_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_21 = arg345_1 = None
        add_268 = torch.ops.aten.add.Tensor(arg347_1, 1e-05);  arg347_1 = None
        sqrt_123 = torch.ops.aten.sqrt.default(add_268);  add_268 = None
        reciprocal_123 = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_428 = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_984 = torch.ops.aten.unsqueeze.default(arg346_1, -1);  arg346_1 = None
        unsqueeze_985 = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        unsqueeze_986 = torch.ops.aten.unsqueeze.default(mul_428, -1);  mul_428 = None
        unsqueeze_987 = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        sub_123 = torch.ops.aten.sub.Tensor(convolution_149, unsqueeze_985);  convolution_149 = unsqueeze_985 = None
        mul_429 = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988 = torch.ops.aten.unsqueeze.default(arg348_1, -1);  arg348_1 = None
        unsqueeze_989 = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_430 = torch.ops.aten.mul.Tensor(mul_429, unsqueeze_989);  mul_429 = unsqueeze_989 = None
        unsqueeze_990 = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_991 = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_269 = torch.ops.aten.add.Tensor(mul_430, unsqueeze_991);  mul_430 = unsqueeze_991 = None
        sigmoid_59 = torch.ops.aten.sigmoid.default(add_269)
        mul_431 = torch.ops.aten.mul.Tensor(add_269, sigmoid_59);  add_269 = sigmoid_59 = None
        mean_27 = torch.ops.aten.mean.dim(mul_431, [-1, -2], True);  mul_431 = None
        view_1 = torch.ops.aten.view.default(mean_27, [8, 1280]);  mean_27 = None
        permute_1 = torch.ops.aten.permute.default(arg350_1, [1, 0]);  arg350_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg351_1, view_1, permute_1);  arg351_1 = view_1 = permute_1 = None
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
    buf11 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf11, (16, 32, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf12, (16,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf13, (16,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf14, (16,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf15, (16,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf16, (96, 16, 1, 1), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf17, (96,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf18, (96,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf19, (96,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf20, (96,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 3456, device=device(type='cuda', index=0))
    reader.tensor(buf21, (96, 1, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf22, (96,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf23, (96,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf24, (96,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf25, (96,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 10368, device=device(type='cuda', index=0))
    reader.tensor(buf26, (27, 96, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 108, device=device(type='cuda', index=0))
    reader.tensor(buf27, (27,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 108, device=device(type='cuda', index=0))
    reader.tensor(buf28, (27,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 108, device=device(type='cuda', index=0))
    reader.tensor(buf29, (27,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 108, device=device(type='cuda', index=0))
    reader.tensor(buf30, (27,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 17496, device=device(type='cuda', index=0))
    reader.tensor(buf31, (162, 27, 1, 1), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf32, (162,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf33, (162,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf34, (162,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf35, (162,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 5832, device=device(type='cuda', index=0))
    reader.tensor(buf36, (162, 1, 3, 3), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf37, (162,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf38, (162,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf39, (162,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf40, (162,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 24624, device=device(type='cuda', index=0))
    reader.tensor(buf41, (38, 162, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 152, device=device(type='cuda', index=0))
    reader.tensor(buf42, (38,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 152, device=device(type='cuda', index=0))
    reader.tensor(buf43, (38,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 152, device=device(type='cuda', index=0))
    reader.tensor(buf44, (38,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 152, device=device(type='cuda', index=0))
    reader.tensor(buf45, (38,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 34656, device=device(type='cuda', index=0))
    reader.tensor(buf46, (228, 38, 1, 1), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 912, device=device(type='cuda', index=0))
    reader.tensor(buf47, (228,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 912, device=device(type='cuda', index=0))
    reader.tensor(buf48, (228,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 912, device=device(type='cuda', index=0))
    reader.tensor(buf49, (228,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 912, device=device(type='cuda', index=0))
    reader.tensor(buf50, (228,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 8208, device=device(type='cuda', index=0))
    reader.tensor(buf51, (228, 1, 3, 3), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 912, device=device(type='cuda', index=0))
    reader.tensor(buf52, (228,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 912, device=device(type='cuda', index=0))
    reader.tensor(buf53, (228,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 912, device=device(type='cuda', index=0))
    reader.tensor(buf54, (228,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 912, device=device(type='cuda', index=0))
    reader.tensor(buf55, (228,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 17328, device=device(type='cuda', index=0))
    reader.tensor(buf56, (19, 228, 1, 1), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 76, device=device(type='cuda', index=0))
    reader.tensor(buf57, (19,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 76, device=device(type='cuda', index=0))
    reader.tensor(buf58, (19,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 76, device=device(type='cuda', index=0))
    reader.tensor(buf59, (19,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 76, device=device(type='cuda', index=0))
    reader.tensor(buf60, (19,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 76, device=device(type='cuda', index=0))
    reader.tensor(buf61, (19,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 17328, device=device(type='cuda', index=0))
    reader.tensor(buf62, (228, 19, 1, 1), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 912, device=device(type='cuda', index=0))
    reader.tensor(buf63, (228,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 45600, device=device(type='cuda', index=0))
    reader.tensor(buf64, (50, 228, 1, 1), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 200, device=device(type='cuda', index=0))
    reader.tensor(buf65, (50,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 200, device=device(type='cuda', index=0))
    reader.tensor(buf66, (50,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 200, device=device(type='cuda', index=0))
    reader.tensor(buf67, (50,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 200, device=device(type='cuda', index=0))
    reader.tensor(buf68, (50,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 60000, device=device(type='cuda', index=0))
    reader.tensor(buf69, (300, 50, 1, 1), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1200, device=device(type='cuda', index=0))
    reader.tensor(buf70, (300,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 1200, device=device(type='cuda', index=0))
    reader.tensor(buf71, (300,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1200, device=device(type='cuda', index=0))
    reader.tensor(buf72, (300,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1200, device=device(type='cuda', index=0))
    reader.tensor(buf73, (300,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 10800, device=device(type='cuda', index=0))
    reader.tensor(buf74, (300, 1, 3, 3), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1200, device=device(type='cuda', index=0))
    reader.tensor(buf75, (300,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1200, device=device(type='cuda', index=0))
    reader.tensor(buf76, (300,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1200, device=device(type='cuda', index=0))
    reader.tensor(buf77, (300,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1200, device=device(type='cuda', index=0))
    reader.tensor(buf78, (300,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 30000, device=device(type='cuda', index=0))
    reader.tensor(buf79, (25, 300, 1, 1), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 100, device=device(type='cuda', index=0))
    reader.tensor(buf80, (25,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 100, device=device(type='cuda', index=0))
    reader.tensor(buf81, (25,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 100, device=device(type='cuda', index=0))
    reader.tensor(buf82, (25,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 100, device=device(type='cuda', index=0))
    reader.tensor(buf83, (25,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 100, device=device(type='cuda', index=0))
    reader.tensor(buf84, (25,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 30000, device=device(type='cuda', index=0))
    reader.tensor(buf85, (300, 25, 1, 1), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 1200, device=device(type='cuda', index=0))
    reader.tensor(buf86, (300,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 73200, device=device(type='cuda', index=0))
    reader.tensor(buf87, (61, 300, 1, 1), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 244, device=device(type='cuda', index=0))
    reader.tensor(buf88, (61,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 244, device=device(type='cuda', index=0))
    reader.tensor(buf89, (61,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 244, device=device(type='cuda', index=0))
    reader.tensor(buf90, (61,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 244, device=device(type='cuda', index=0))
    reader.tensor(buf91, (61,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 89304, device=device(type='cuda', index=0))
    reader.tensor(buf92, (366, 61, 1, 1), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1464, device=device(type='cuda', index=0))
    reader.tensor(buf93, (366,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1464, device=device(type='cuda', index=0))
    reader.tensor(buf94, (366,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 1464, device=device(type='cuda', index=0))
    reader.tensor(buf95, (366,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1464, device=device(type='cuda', index=0))
    reader.tensor(buf96, (366,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 13176, device=device(type='cuda', index=0))
    reader.tensor(buf97, (366, 1, 3, 3), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1464, device=device(type='cuda', index=0))
    reader.tensor(buf98, (366,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1464, device=device(type='cuda', index=0))
    reader.tensor(buf99, (366,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1464, device=device(type='cuda', index=0))
    reader.tensor(buf100, (366,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1464, device=device(type='cuda', index=0))
    reader.tensor(buf101, (366,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 43920, device=device(type='cuda', index=0))
    reader.tensor(buf102, (30, 366, 1, 1), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 120, device=device(type='cuda', index=0))
    reader.tensor(buf103, (30,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 120, device=device(type='cuda', index=0))
    reader.tensor(buf104, (30,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 120, device=device(type='cuda', index=0))
    reader.tensor(buf105, (30,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 120, device=device(type='cuda', index=0))
    reader.tensor(buf106, (30,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 120, device=device(type='cuda', index=0))
    reader.tensor(buf107, (30,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 43920, device=device(type='cuda', index=0))
    reader.tensor(buf108, (366, 30, 1, 1), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 1464, device=device(type='cuda', index=0))
    reader.tensor(buf109, (366,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 105408, device=device(type='cuda', index=0))
    reader.tensor(buf110, (72, 366, 1, 1), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf111, (72,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf112, (72,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf113, (72,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf114, (72,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 124416, device=device(type='cuda', index=0))
    reader.tensor(buf115, (432, 72, 1, 1), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf116, (432,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf117, (432,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf118, (432,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf119, (432,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 15552, device=device(type='cuda', index=0))
    reader.tensor(buf120, (432, 1, 3, 3), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf121, (432,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf122, (432,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf123, (432,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf124, (432,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf125, (36, 432, 1, 1), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf126, (36,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf127, (36,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf128, (36,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf129, (36,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 144, device=device(type='cuda', index=0))
    reader.tensor(buf130, (36,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 62208, device=device(type='cuda', index=0))
    reader.tensor(buf131, (432, 36, 1, 1), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf132, (432,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 145152, device=device(type='cuda', index=0))
    reader.tensor(buf133, (84, 432, 1, 1), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 336, device=device(type='cuda', index=0))
    reader.tensor(buf134, (84,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 336, device=device(type='cuda', index=0))
    reader.tensor(buf135, (84,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 336, device=device(type='cuda', index=0))
    reader.tensor(buf136, (84,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 336, device=device(type='cuda', index=0))
    reader.tensor(buf137, (84,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 169344, device=device(type='cuda', index=0))
    reader.tensor(buf138, (504, 84, 1, 1), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf139, (504,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf140, (504,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf141, (504,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf142, (504,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 18144, device=device(type='cuda', index=0))
    reader.tensor(buf143, (504, 1, 3, 3), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf144, (504,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf145, (504,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf146, (504,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf147, (504,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf148, (42, 504, 1, 1), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 168, device=device(type='cuda', index=0))
    reader.tensor(buf149, (42,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 168, device=device(type='cuda', index=0))
    reader.tensor(buf150, (42,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 168, device=device(type='cuda', index=0))
    reader.tensor(buf151, (42,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 168, device=device(type='cuda', index=0))
    reader.tensor(buf152, (42,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 168, device=device(type='cuda', index=0))
    reader.tensor(buf153, (42,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 84672, device=device(type='cuda', index=0))
    reader.tensor(buf154, (504, 42, 1, 1), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2016, device=device(type='cuda', index=0))
    reader.tensor(buf155, (504,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 191520, device=device(type='cuda', index=0))
    reader.tensor(buf156, (95, 504, 1, 1), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 380, device=device(type='cuda', index=0))
    reader.tensor(buf157, (95,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 380, device=device(type='cuda', index=0))
    reader.tensor(buf158, (95,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 380, device=device(type='cuda', index=0))
    reader.tensor(buf159, (95,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 380, device=device(type='cuda', index=0))
    reader.tensor(buf160, (95,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 216600, device=device(type='cuda', index=0))
    reader.tensor(buf161, (570, 95, 1, 1), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 2280, device=device(type='cuda', index=0))
    reader.tensor(buf162, (570,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 2280, device=device(type='cuda', index=0))
    reader.tensor(buf163, (570,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2280, device=device(type='cuda', index=0))
    reader.tensor(buf164, (570,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2280, device=device(type='cuda', index=0))
    reader.tensor(buf165, (570,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 20520, device=device(type='cuda', index=0))
    reader.tensor(buf166, (570, 1, 3, 3), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 2280, device=device(type='cuda', index=0))
    reader.tensor(buf167, (570,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 2280, device=device(type='cuda', index=0))
    reader.tensor(buf168, (570,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2280, device=device(type='cuda', index=0))
    reader.tensor(buf169, (570,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 2280, device=device(type='cuda', index=0))
    reader.tensor(buf170, (570,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 107160, device=device(type='cuda', index=0))
    reader.tensor(buf171, (47, 570, 1, 1), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 188, device=device(type='cuda', index=0))
    reader.tensor(buf172, (47,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 188, device=device(type='cuda', index=0))
    reader.tensor(buf173, (47,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 188, device=device(type='cuda', index=0))
    reader.tensor(buf174, (47,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 188, device=device(type='cuda', index=0))
    reader.tensor(buf175, (47,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 188, device=device(type='cuda', index=0))
    reader.tensor(buf176, (47,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 107160, device=device(type='cuda', index=0))
    reader.tensor(buf177, (570, 47, 1, 1), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 2280, device=device(type='cuda', index=0))
    reader.tensor(buf178, (570,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 241680, device=device(type='cuda', index=0))
    reader.tensor(buf179, (106, 570, 1, 1), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 424, device=device(type='cuda', index=0))
    reader.tensor(buf180, (106,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 424, device=device(type='cuda', index=0))
    reader.tensor(buf181, (106,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 424, device=device(type='cuda', index=0))
    reader.tensor(buf182, (106,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 424, device=device(type='cuda', index=0))
    reader.tensor(buf183, (106,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 269664, device=device(type='cuda', index=0))
    reader.tensor(buf184, (636, 106, 1, 1), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2544, device=device(type='cuda', index=0))
    reader.tensor(buf185, (636,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 2544, device=device(type='cuda', index=0))
    reader.tensor(buf186, (636,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2544, device=device(type='cuda', index=0))
    reader.tensor(buf187, (636,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 2544, device=device(type='cuda', index=0))
    reader.tensor(buf188, (636,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 22896, device=device(type='cuda', index=0))
    reader.tensor(buf189, (636, 1, 3, 3), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 2544, device=device(type='cuda', index=0))
    reader.tensor(buf190, (636,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 2544, device=device(type='cuda', index=0))
    reader.tensor(buf191, (636,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 2544, device=device(type='cuda', index=0))
    reader.tensor(buf192, (636,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 2544, device=device(type='cuda', index=0))
    reader.tensor(buf193, (636,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 134832, device=device(type='cuda', index=0))
    reader.tensor(buf194, (53, 636, 1, 1), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 212, device=device(type='cuda', index=0))
    reader.tensor(buf195, (53,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 212, device=device(type='cuda', index=0))
    reader.tensor(buf196, (53,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 212, device=device(type='cuda', index=0))
    reader.tensor(buf197, (53,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 212, device=device(type='cuda', index=0))
    reader.tensor(buf198, (53,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 212, device=device(type='cuda', index=0))
    reader.tensor(buf199, (53,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 134832, device=device(type='cuda', index=0))
    reader.tensor(buf200, (636, 53, 1, 1), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 2544, device=device(type='cuda', index=0))
    reader.tensor(buf201, (636,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 297648, device=device(type='cuda', index=0))
    reader.tensor(buf202, (117, 636, 1, 1), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 468, device=device(type='cuda', index=0))
    reader.tensor(buf203, (117,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 468, device=device(type='cuda', index=0))
    reader.tensor(buf204, (117,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 468, device=device(type='cuda', index=0))
    reader.tensor(buf205, (117,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 468, device=device(type='cuda', index=0))
    reader.tensor(buf206, (117,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 328536, device=device(type='cuda', index=0))
    reader.tensor(buf207, (702, 117, 1, 1), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 2808, device=device(type='cuda', index=0))
    reader.tensor(buf208, (702,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 2808, device=device(type='cuda', index=0))
    reader.tensor(buf209, (702,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 2808, device=device(type='cuda', index=0))
    reader.tensor(buf210, (702,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2808, device=device(type='cuda', index=0))
    reader.tensor(buf211, (702,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 25272, device=device(type='cuda', index=0))
    reader.tensor(buf212, (702, 1, 3, 3), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 2808, device=device(type='cuda', index=0))
    reader.tensor(buf213, (702,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 2808, device=device(type='cuda', index=0))
    reader.tensor(buf214, (702,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 2808, device=device(type='cuda', index=0))
    reader.tensor(buf215, (702,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 2808, device=device(type='cuda', index=0))
    reader.tensor(buf216, (702,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 162864, device=device(type='cuda', index=0))
    reader.tensor(buf217, (58, 702, 1, 1), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 232, device=device(type='cuda', index=0))
    reader.tensor(buf218, (58,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 232, device=device(type='cuda', index=0))
    reader.tensor(buf219, (58,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 232, device=device(type='cuda', index=0))
    reader.tensor(buf220, (58,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 232, device=device(type='cuda', index=0))
    reader.tensor(buf221, (58,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 232, device=device(type='cuda', index=0))
    reader.tensor(buf222, (58,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 162864, device=device(type='cuda', index=0))
    reader.tensor(buf223, (702, 58, 1, 1), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 2808, device=device(type='cuda', index=0))
    reader.tensor(buf224, (702,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 359424, device=device(type='cuda', index=0))
    reader.tensor(buf225, (128, 702, 1, 1), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf226, (128,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf227, (128,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf228, (128,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf229, (128,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 393216, device=device(type='cuda', index=0))
    reader.tensor(buf230, (768, 128, 1, 1), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf231, (768,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf232, (768,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf233, (768,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf234, (768,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf235, (768, 1, 3, 3), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf236, (768,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf237, (768,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf238, (768,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf239, (768,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf240, (64, 768, 1, 1), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf241, (64,), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf242, (64,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf243, (64,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf244, (64,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf245, (64,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf246, (768, 64, 1, 1), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf247, (768,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 430080, device=device(type='cuda', index=0))
    reader.tensor(buf248, (140, 768, 1, 1), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 560, device=device(type='cuda', index=0))
    reader.tensor(buf249, (140,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 560, device=device(type='cuda', index=0))
    reader.tensor(buf250, (140,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 560, device=device(type='cuda', index=0))
    reader.tensor(buf251, (140,), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 560, device=device(type='cuda', index=0))
    reader.tensor(buf252, (140,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 470400, device=device(type='cuda', index=0))
    reader.tensor(buf253, (840, 140, 1, 1), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 3360, device=device(type='cuda', index=0))
    reader.tensor(buf254, (840,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 3360, device=device(type='cuda', index=0))
    reader.tensor(buf255, (840,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 3360, device=device(type='cuda', index=0))
    reader.tensor(buf256, (840,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 3360, device=device(type='cuda', index=0))
    reader.tensor(buf257, (840,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 30240, device=device(type='cuda', index=0))
    reader.tensor(buf258, (840, 1, 3, 3), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 3360, device=device(type='cuda', index=0))
    reader.tensor(buf259, (840,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 3360, device=device(type='cuda', index=0))
    reader.tensor(buf260, (840,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 3360, device=device(type='cuda', index=0))
    reader.tensor(buf261, (840,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 3360, device=device(type='cuda', index=0))
    reader.tensor(buf262, (840,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 235200, device=device(type='cuda', index=0))
    reader.tensor(buf263, (70, 840, 1, 1), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 280, device=device(type='cuda', index=0))
    reader.tensor(buf264, (70,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 280, device=device(type='cuda', index=0))
    reader.tensor(buf265, (70,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 280, device=device(type='cuda', index=0))
    reader.tensor(buf266, (70,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 280, device=device(type='cuda', index=0))
    reader.tensor(buf267, (70,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 280, device=device(type='cuda', index=0))
    reader.tensor(buf268, (70,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 235200, device=device(type='cuda', index=0))
    reader.tensor(buf269, (840, 70, 1, 1), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 3360, device=device(type='cuda', index=0))
    reader.tensor(buf270, (840,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 507360, device=device(type='cuda', index=0))
    reader.tensor(buf271, (151, 840, 1, 1), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 604, device=device(type='cuda', index=0))
    reader.tensor(buf272, (151,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 604, device=device(type='cuda', index=0))
    reader.tensor(buf273, (151,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 604, device=device(type='cuda', index=0))
    reader.tensor(buf274, (151,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 604, device=device(type='cuda', index=0))
    reader.tensor(buf275, (151,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 547224, device=device(type='cuda', index=0))
    reader.tensor(buf276, (906, 151, 1, 1), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 3624, device=device(type='cuda', index=0))
    reader.tensor(buf277, (906,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 3624, device=device(type='cuda', index=0))
    reader.tensor(buf278, (906,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 3624, device=device(type='cuda', index=0))
    reader.tensor(buf279, (906,), is_leaf=True)  # arg279_1
    buf280 = reader.storage(None, 3624, device=device(type='cuda', index=0))
    reader.tensor(buf280, (906,), is_leaf=True)  # arg280_1
    buf281 = reader.storage(None, 32616, device=device(type='cuda', index=0))
    reader.tensor(buf281, (906, 1, 3, 3), is_leaf=True)  # arg281_1
    buf282 = reader.storage(None, 3624, device=device(type='cuda', index=0))
    reader.tensor(buf282, (906,), is_leaf=True)  # arg282_1
    buf283 = reader.storage(None, 3624, device=device(type='cuda', index=0))
    reader.tensor(buf283, (906,), is_leaf=True)  # arg283_1
    buf284 = reader.storage(None, 3624, device=device(type='cuda', index=0))
    reader.tensor(buf284, (906,), is_leaf=True)  # arg284_1
    buf285 = reader.storage(None, 3624, device=device(type='cuda', index=0))
    reader.tensor(buf285, (906,), is_leaf=True)  # arg285_1
    buf286 = reader.storage(None, 271800, device=device(type='cuda', index=0))
    reader.tensor(buf286, (75, 906, 1, 1), is_leaf=True)  # arg286_1
    buf287 = reader.storage(None, 300, device=device(type='cuda', index=0))
    reader.tensor(buf287, (75,), is_leaf=True)  # arg287_1
    buf288 = reader.storage(None, 300, device=device(type='cuda', index=0))
    reader.tensor(buf288, (75,), is_leaf=True)  # arg288_1
    buf289 = reader.storage(None, 300, device=device(type='cuda', index=0))
    reader.tensor(buf289, (75,), is_leaf=True)  # arg289_1
    buf290 = reader.storage(None, 300, device=device(type='cuda', index=0))
    reader.tensor(buf290, (75,), is_leaf=True)  # arg290_1
    buf291 = reader.storage(None, 300, device=device(type='cuda', index=0))
    reader.tensor(buf291, (75,), is_leaf=True)  # arg291_1
    buf292 = reader.storage(None, 271800, device=device(type='cuda', index=0))
    reader.tensor(buf292, (906, 75, 1, 1), is_leaf=True)  # arg292_1
    buf293 = reader.storage(None, 3624, device=device(type='cuda', index=0))
    reader.tensor(buf293, (906,), is_leaf=True)  # arg293_1
    buf294 = reader.storage(None, 587088, device=device(type='cuda', index=0))
    reader.tensor(buf294, (162, 906, 1, 1), is_leaf=True)  # arg294_1
    buf295 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf295, (162,), is_leaf=True)  # arg295_1
    buf296 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf296, (162,), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf297, (162,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 648, device=device(type='cuda', index=0))
    reader.tensor(buf298, (162,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 629856, device=device(type='cuda', index=0))
    reader.tensor(buf299, (972, 162, 1, 1), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf300, (972,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf301, (972,), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf302, (972,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf303, (972,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 34992, device=device(type='cuda', index=0))
    reader.tensor(buf304, (972, 1, 3, 3), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf305, (972,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf306, (972,), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf307, (972,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf308, (972,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 314928, device=device(type='cuda', index=0))
    reader.tensor(buf309, (81, 972, 1, 1), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 324, device=device(type='cuda', index=0))
    reader.tensor(buf310, (81,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 324, device=device(type='cuda', index=0))
    reader.tensor(buf311, (81,), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 324, device=device(type='cuda', index=0))
    reader.tensor(buf312, (81,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 324, device=device(type='cuda', index=0))
    reader.tensor(buf313, (81,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 324, device=device(type='cuda', index=0))
    reader.tensor(buf314, (81,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 314928, device=device(type='cuda', index=0))
    reader.tensor(buf315, (972, 81, 1, 1), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 3888, device=device(type='cuda', index=0))
    reader.tensor(buf316, (972,), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 676512, device=device(type='cuda', index=0))
    reader.tensor(buf317, (174, 972, 1, 1), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 696, device=device(type='cuda', index=0))
    reader.tensor(buf318, (174,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 696, device=device(type='cuda', index=0))
    reader.tensor(buf319, (174,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 696, device=device(type='cuda', index=0))
    reader.tensor(buf320, (174,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 696, device=device(type='cuda', index=0))
    reader.tensor(buf321, (174,), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 726624, device=device(type='cuda', index=0))
    reader.tensor(buf322, (1044, 174, 1, 1), is_leaf=True)  # arg322_1
    buf323 = reader.storage(None, 4176, device=device(type='cuda', index=0))
    reader.tensor(buf323, (1044,), is_leaf=True)  # arg323_1
    buf324 = reader.storage(None, 4176, device=device(type='cuda', index=0))
    reader.tensor(buf324, (1044,), is_leaf=True)  # arg324_1
    buf325 = reader.storage(None, 4176, device=device(type='cuda', index=0))
    reader.tensor(buf325, (1044,), is_leaf=True)  # arg325_1
    buf326 = reader.storage(None, 4176, device=device(type='cuda', index=0))
    reader.tensor(buf326, (1044,), is_leaf=True)  # arg326_1
    buf327 = reader.storage(None, 37584, device=device(type='cuda', index=0))
    reader.tensor(buf327, (1044, 1, 3, 3), is_leaf=True)  # arg327_1
    buf328 = reader.storage(None, 4176, device=device(type='cuda', index=0))
    reader.tensor(buf328, (1044,), is_leaf=True)  # arg328_1
    buf329 = reader.storage(None, 4176, device=device(type='cuda', index=0))
    reader.tensor(buf329, (1044,), is_leaf=True)  # arg329_1
    buf330 = reader.storage(None, 4176, device=device(type='cuda', index=0))
    reader.tensor(buf330, (1044,), is_leaf=True)  # arg330_1
    buf331 = reader.storage(None, 4176, device=device(type='cuda', index=0))
    reader.tensor(buf331, (1044,), is_leaf=True)  # arg331_1
    buf332 = reader.storage(None, 363312, device=device(type='cuda', index=0))
    reader.tensor(buf332, (87, 1044, 1, 1), is_leaf=True)  # arg332_1
    buf333 = reader.storage(None, 348, device=device(type='cuda', index=0))
    reader.tensor(buf333, (87,), is_leaf=True)  # arg333_1
    buf334 = reader.storage(None, 348, device=device(type='cuda', index=0))
    reader.tensor(buf334, (87,), is_leaf=True)  # arg334_1
    buf335 = reader.storage(None, 348, device=device(type='cuda', index=0))
    reader.tensor(buf335, (87,), is_leaf=True)  # arg335_1
    buf336 = reader.storage(None, 348, device=device(type='cuda', index=0))
    reader.tensor(buf336, (87,), is_leaf=True)  # arg336_1
    buf337 = reader.storage(None, 348, device=device(type='cuda', index=0))
    reader.tensor(buf337, (87,), is_leaf=True)  # arg337_1
    buf338 = reader.storage(None, 363312, device=device(type='cuda', index=0))
    reader.tensor(buf338, (1044, 87, 1, 1), is_leaf=True)  # arg338_1
    buf339 = reader.storage(None, 4176, device=device(type='cuda', index=0))
    reader.tensor(buf339, (1044,), is_leaf=True)  # arg339_1
    buf340 = reader.storage(None, 772560, device=device(type='cuda', index=0))
    reader.tensor(buf340, (185, 1044, 1, 1), is_leaf=True)  # arg340_1
    buf341 = reader.storage(None, 740, device=device(type='cuda', index=0))
    reader.tensor(buf341, (185,), is_leaf=True)  # arg341_1
    buf342 = reader.storage(None, 740, device=device(type='cuda', index=0))
    reader.tensor(buf342, (185,), is_leaf=True)  # arg342_1
    buf343 = reader.storage(None, 740, device=device(type='cuda', index=0))
    reader.tensor(buf343, (185,), is_leaf=True)  # arg343_1
    buf344 = reader.storage(None, 740, device=device(type='cuda', index=0))
    reader.tensor(buf344, (185,), is_leaf=True)  # arg344_1
    buf345 = reader.storage(None, 947200, device=device(type='cuda', index=0))
    reader.tensor(buf345, (1280, 185, 1, 1), is_leaf=True)  # arg345_1
    buf346 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf346, (1280,), is_leaf=True)  # arg346_1
    buf347 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf347, (1280,), is_leaf=True)  # arg347_1
    buf348 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf348, (1280,), is_leaf=True)  # arg348_1
    buf349 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf349, (1280,), is_leaf=True)  # arg349_1
    buf350 = reader.storage(None, 5120000, device=device(type='cuda', index=0))
    reader.tensor(buf350, (1000, 1280), is_leaf=True)  # arg350_1
    buf351 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf351, (1000,), is_leaf=True)  # arg351_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)