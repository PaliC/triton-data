
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1):
        convolution_64 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_142 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_64 = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_64 = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_192 = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64 = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_513);  convolution_64 = unsqueeze_513 = None
        mul_193 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_143 = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
        relu_43 = torch.ops.aten.relu.default(add_143);  add_143 = None
        convolution_65 = torch.ops.aten.convolution.default(relu_43, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  relu_43 = arg6_1 = None
        add_144 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_65 = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_65 = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_195 = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65 = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_521);  convolution_65 = unsqueeze_521 = None
        mul_196 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_145 = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
        relu_44 = torch.ops.aten.relu.default(add_145);  add_145 = None
        convolution_66 = torch.ops.aten.convolution.default(relu_44, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_44 = arg11_1 = None
        add_146 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_66 = torch.ops.aten.sqrt.default(add_146);  add_146 = None
        reciprocal_66 = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_198 = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66 = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_529);  convolution_66 = unsqueeze_529 = None
        mul_199 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_147 = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
        convolution_67 = torch.ops.aten.convolution.default(add_147, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_147 = arg16_1 = None
        add_148 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_67 = torch.ops.aten.sqrt.default(add_148);  add_148 = None
        reciprocal_67 = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_201 = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_537);  convolution_67 = unsqueeze_537 = None
        mul_202 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_543 = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_149 = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
        relu_45 = torch.ops.aten.relu.default(add_149);  add_149 = None
        convolution_68 = torch.ops.aten.convolution.default(relu_45, arg21_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 48);  relu_45 = arg21_1 = None
        add_150 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_68 = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_68 = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_204 = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_545 = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68 = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_545);  convolution_68 = unsqueeze_545 = None
        mul_205 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_206 = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_151 = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
        relu_46 = torch.ops.aten.relu.default(add_151);  add_151 = None
        convolution_69 = torch.ops.aten.convolution.default(relu_46, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_46 = arg26_1 = None
        add_152 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_69 = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_69 = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_207 = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69 = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_553);  convolution_69 = unsqueeze_553 = None
        mul_208 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_153 = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
        convolution_70 = torch.ops.aten.convolution.default(add_153, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg31_1 = None
        add_154 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_70 = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_70 = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_210 = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70 = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_561);  convolution_70 = unsqueeze_561 = None
        mul_211 = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_155 = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
        relu_47 = torch.ops.aten.relu.default(add_155);  add_155 = None
        convolution_71 = torch.ops.aten.convolution.default(relu_47, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72);  relu_47 = arg36_1 = None
        add_156 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_71 = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_71 = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_213 = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71 = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_569);  convolution_71 = unsqueeze_569 = None
        mul_214 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_215 = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_157 = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
        relu_48 = torch.ops.aten.relu.default(add_157);  add_157 = None
        convolution_72 = torch.ops.aten.convolution.default(relu_48, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_48 = arg41_1 = None
        add_158 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_72 = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_72 = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_216 = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72 = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_577);  convolution_72 = unsqueeze_577 = None
        mul_217 = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_218 = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_159 = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
        add_160 = torch.ops.aten.add.Tensor(add_159, add_153);  add_159 = add_153 = None
        convolution_73 = torch.ops.aten.convolution.default(add_160, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg46_1 = None
        add_161 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_73 = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_73 = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_219 = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73 = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_585);  convolution_73 = unsqueeze_585 = None
        mul_220 = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_162 = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
        relu_49 = torch.ops.aten.relu.default(add_162);  add_162 = None
        convolution_74 = torch.ops.aten.convolution.default(relu_49, arg51_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 72);  relu_49 = arg51_1 = None
        add_163 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_74 = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_74 = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_222 = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74 = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_593);  convolution_74 = unsqueeze_593 = None
        mul_223 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_224 = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_164 = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
        relu_50 = torch.ops.aten.relu.default(add_164);  add_164 = None
        convolution_75 = torch.ops.aten.convolution.default(relu_50, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_50 = arg56_1 = None
        add_165 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_75 = torch.ops.aten.sqrt.default(add_165);  add_165 = None
        reciprocal_75 = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_225 = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75 = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_601);  convolution_75 = unsqueeze_601 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_605 = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_227 = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_166 = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
        add_167 = torch.ops.aten.add.Tensor(add_166, add_160);  add_166 = add_160 = None
        convolution_76 = torch.ops.aten.convolution.default(add_167, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_167 = arg61_1 = None
        add_168 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_76 = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_76 = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_228 = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_611 = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76 = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_609);  convolution_76 = unsqueeze_609 = None
        mul_229 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_613 = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
        unsqueeze_614 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_615 = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_169 = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
        relu_51 = torch.ops.aten.relu.default(add_169);  add_169 = None
        convolution_77 = torch.ops.aten.convolution.default(relu_51, arg66_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 144);  relu_51 = arg66_1 = None
        add_170 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_77 = torch.ops.aten.sqrt.default(add_170);  add_170 = None
        reciprocal_77 = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_231 = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_617 = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618 = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
        unsqueeze_619 = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77 = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_617);  convolution_77 = unsqueeze_617 = None
        mul_232 = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_623 = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_171 = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
        relu_52 = torch.ops.aten.relu.default(add_171);  add_171 = None
        convolution_78 = torch.ops.aten.convolution.default(relu_52, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_52 = arg71_1 = None
        add_172 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_78 = torch.ops.aten.sqrt.default(add_172);  add_172 = None
        reciprocal_78 = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_234 = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_625 = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626 = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
        unsqueeze_627 = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78 = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_625);  convolution_78 = unsqueeze_625 = None
        mul_235 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_629 = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
        unsqueeze_630 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_631 = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_173 = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
        convolution_79 = torch.ops.aten.convolution.default(add_173, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg76_1 = None
        add_174 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_79 = torch.ops.aten.sqrt.default(add_174);  add_174 = None
        reciprocal_79 = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_237 = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_635 = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79 = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_633);  convolution_79 = unsqueeze_633 = None
        mul_238 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_637 = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
        unsqueeze_638 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_639 = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_175 = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
        relu_53 = torch.ops.aten.relu.default(add_175);  add_175 = None
        convolution_80 = torch.ops.aten.convolution.default(relu_53, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  relu_53 = arg81_1 = None
        add_176 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_80 = torch.ops.aten.sqrt.default(add_176);  add_176 = None
        reciprocal_80 = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_240 = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_641 = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80 = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_641);  convolution_80 = unsqueeze_641 = None
        mul_241 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_177 = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
        relu_54 = torch.ops.aten.relu.default(add_177);  add_177 = None
        convolution_81 = torch.ops.aten.convolution.default(relu_54, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_54 = arg86_1 = None
        add_178 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_81 = torch.ops.aten.sqrt.default(add_178);  add_178 = None
        reciprocal_81 = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_243 = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_651 = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81 = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_649);  convolution_81 = unsqueeze_649 = None
        mul_244 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_653 = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_179 = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
        add_180 = torch.ops.aten.add.Tensor(add_179, add_173);  add_179 = add_173 = None
        convolution_82 = torch.ops.aten.convolution.default(add_180, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg91_1 = None
        add_181 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_82 = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_82 = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_246 = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
        unsqueeze_659 = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82 = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_657);  convolution_82 = unsqueeze_657 = None
        mul_247 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_661 = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_661);  mul_247 = unsqueeze_661 = None
        unsqueeze_662 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_663 = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_182 = torch.ops.aten.add.Tensor(mul_248, unsqueeze_663);  mul_248 = unsqueeze_663 = None
        relu_55 = torch.ops.aten.relu.default(add_182);  add_182 = None
        convolution_83 = torch.ops.aten.convolution.default(relu_55, arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  relu_55 = arg96_1 = None
        add_183 = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_83 = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_83 = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_249 = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_665 = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666 = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
        unsqueeze_667 = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83 = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_665);  convolution_83 = unsqueeze_665 = None
        mul_250 = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_669);  mul_250 = unsqueeze_669 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_671 = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_184 = torch.ops.aten.add.Tensor(mul_251, unsqueeze_671);  mul_251 = unsqueeze_671 = None
        relu_56 = torch.ops.aten.relu.default(add_184);  add_184 = None
        convolution_84 = torch.ops.aten.convolution.default(relu_56, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_56 = arg101_1 = None
        add_185 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_84 = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_84 = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_252 = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_673 = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674 = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
        unsqueeze_675 = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84 = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_673);  convolution_84 = unsqueeze_673 = None
        mul_253 = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_677 = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_254 = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_677);  mul_253 = unsqueeze_677 = None
        unsqueeze_678 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_679 = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_186 = torch.ops.aten.add.Tensor(mul_254, unsqueeze_679);  mul_254 = unsqueeze_679 = None
        add_187 = torch.ops.aten.add.Tensor(add_186, add_180);  add_186 = add_180 = None
        convolution_85 = torch.ops.aten.convolution.default(add_187, arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg106_1 = None
        add_188 = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_85 = torch.ops.aten.sqrt.default(add_188);  add_188 = None
        reciprocal_85 = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_255 = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85 = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_681);  convolution_85 = unsqueeze_681 = None
        mul_256 = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_685);  mul_256 = unsqueeze_685 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_687 = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_189 = torch.ops.aten.add.Tensor(mul_257, unsqueeze_687);  mul_257 = unsqueeze_687 = None
        relu_57 = torch.ops.aten.relu.default(add_189);  add_189 = None
        convolution_86 = torch.ops.aten.convolution.default(relu_57, arg111_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 120);  relu_57 = arg111_1 = None
        add_190 = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_86 = torch.ops.aten.sqrt.default(add_190);  add_190 = None
        reciprocal_86 = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_258 = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_689 = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86 = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_689);  convolution_86 = unsqueeze_689 = None
        mul_259 = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_260 = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_693);  mul_259 = unsqueeze_693 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_191 = torch.ops.aten.add.Tensor(mul_260, unsqueeze_695);  mul_260 = unsqueeze_695 = None
        relu_58 = torch.ops.aten.relu.default(add_191);  add_191 = None
        convolution_87 = torch.ops.aten.convolution.default(relu_58, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_58 = arg116_1 = None
        add_192 = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_87 = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_87 = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_261 = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87 = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_697);  convolution_87 = unsqueeze_697 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_701);  mul_262 = unsqueeze_701 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_193 = torch.ops.aten.add.Tensor(mul_263, unsqueeze_703);  mul_263 = unsqueeze_703 = None
        add_194 = torch.ops.aten.add.Tensor(add_193, add_187);  add_193 = add_187 = None
        convolution_88 = torch.ops.aten.convolution.default(add_194, arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_194 = arg121_1 = None
        add_195 = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_88 = torch.ops.aten.sqrt.default(add_195);  add_195 = None
        reciprocal_88 = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_264 = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88 = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_705);  convolution_88 = unsqueeze_705 = None
        mul_265 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_709);  mul_265 = unsqueeze_709 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_196 = torch.ops.aten.add.Tensor(mul_266, unsqueeze_711);  mul_266 = unsqueeze_711 = None
        relu_59 = torch.ops.aten.relu.default(add_196);  add_196 = None
        convolution_89 = torch.ops.aten.convolution.default(relu_59, arg126_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 240);  relu_59 = arg126_1 = None
        add_197 = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_89 = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_89 = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_267 = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89 = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_713);  convolution_89 = unsqueeze_713 = None
        mul_268 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_717);  mul_268 = unsqueeze_717 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_198 = torch.ops.aten.add.Tensor(mul_269, unsqueeze_719);  mul_269 = unsqueeze_719 = None
        relu_60 = torch.ops.aten.relu.default(add_198);  add_198 = None
        convolution_90 = torch.ops.aten.convolution.default(relu_60, arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_60 = arg131_1 = None
        add_199 = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_90 = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_90 = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_270 = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90 = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_721);  convolution_90 = unsqueeze_721 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_725);  mul_271 = unsqueeze_725 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_200 = torch.ops.aten.add.Tensor(mul_272, unsqueeze_727);  mul_272 = unsqueeze_727 = None
        convolution_91 = torch.ops.aten.convolution.default(add_200, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg136_1 = None
        add_201 = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_91 = torch.ops.aten.sqrt.default(add_201);  add_201 = None
        reciprocal_91 = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_273 = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_729);  convolution_91 = unsqueeze_729 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_733);  mul_274 = unsqueeze_733 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_202 = torch.ops.aten.add.Tensor(mul_275, unsqueeze_735);  mul_275 = unsqueeze_735 = None
        relu_61 = torch.ops.aten.relu.default(add_202);  add_202 = None
        convolution_92 = torch.ops.aten.convolution.default(relu_61, arg141_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 240);  relu_61 = arg141_1 = None
        add_203 = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_92 = torch.ops.aten.sqrt.default(add_203);  add_203 = None
        reciprocal_92 = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_276 = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92 = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_737);  convolution_92 = unsqueeze_737 = None
        mul_277 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_741);  mul_277 = unsqueeze_741 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_204 = torch.ops.aten.add.Tensor(mul_278, unsqueeze_743);  mul_278 = unsqueeze_743 = None
        relu_62 = torch.ops.aten.relu.default(add_204);  add_204 = None
        convolution_93 = torch.ops.aten.convolution.default(relu_62, arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_62 = arg146_1 = None
        add_205 = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_93 = torch.ops.aten.sqrt.default(add_205);  add_205 = None
        reciprocal_93 = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_279 = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_745);  convolution_93 = unsqueeze_745 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_749);  mul_280 = unsqueeze_749 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_206 = torch.ops.aten.add.Tensor(mul_281, unsqueeze_751);  mul_281 = unsqueeze_751 = None
        add_207 = torch.ops.aten.add.Tensor(add_206, add_200);  add_206 = add_200 = None
        convolution_94 = torch.ops.aten.convolution.default(add_207, arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg151_1 = None
        add_208 = torch.ops.aten.add.Tensor(arg153_1, 1e-05);  arg153_1 = None
        sqrt_94 = torch.ops.aten.sqrt.default(add_208);  add_208 = None
        reciprocal_94 = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_282 = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94 = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_753);  convolution_94 = unsqueeze_753 = None
        mul_283 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_757);  mul_283 = unsqueeze_757 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_209 = torch.ops.aten.add.Tensor(mul_284, unsqueeze_759);  mul_284 = unsqueeze_759 = None
        relu_63 = torch.ops.aten.relu.default(add_209);  add_209 = None
        convolution_95 = torch.ops.aten.convolution.default(relu_63, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 240);  relu_63 = arg156_1 = None
        add_210 = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_95 = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_95 = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_285 = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95 = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_761);  convolution_95 = unsqueeze_761 = None
        mul_286 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_765);  mul_286 = unsqueeze_765 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_211 = torch.ops.aten.add.Tensor(mul_287, unsqueeze_767);  mul_287 = unsqueeze_767 = None
        relu_64 = torch.ops.aten.relu.default(add_211);  add_211 = None
        convolution_96 = torch.ops.aten.convolution.default(relu_64, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_64 = arg161_1 = None
        add_212 = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_96 = torch.ops.aten.sqrt.default(add_212);  add_212 = None
        reciprocal_96 = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_288 = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_771 = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96 = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_769);  convolution_96 = unsqueeze_769 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772 = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_773 = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_773);  mul_289 = unsqueeze_773 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_213 = torch.ops.aten.add.Tensor(mul_290, unsqueeze_775);  mul_290 = unsqueeze_775 = None
        add_214 = torch.ops.aten.add.Tensor(add_213, add_207);  add_213 = add_207 = None
        convolution_97 = torch.ops.aten.convolution.default(add_214, arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg166_1 = None
        add_215 = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_97 = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_97 = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_291 = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97 = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_777);  convolution_97 = unsqueeze_777 = None
        mul_292 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_781);  mul_292 = unsqueeze_781 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_216 = torch.ops.aten.add.Tensor(mul_293, unsqueeze_783);  mul_293 = unsqueeze_783 = None
        relu_65 = torch.ops.aten.relu.default(add_216);  add_216 = None
        convolution_98 = torch.ops.aten.convolution.default(relu_65, arg171_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 240);  relu_65 = arg171_1 = None
        add_217 = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_98 = torch.ops.aten.sqrt.default(add_217);  add_217 = None
        reciprocal_98 = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_294 = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98 = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_785);  convolution_98 = unsqueeze_785 = None
        mul_295 = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_296 = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_789);  mul_295 = unsqueeze_789 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_218 = torch.ops.aten.add.Tensor(mul_296, unsqueeze_791);  mul_296 = unsqueeze_791 = None
        relu_66 = torch.ops.aten.relu.default(add_218);  add_218 = None
        convolution_99 = torch.ops.aten.convolution.default(relu_66, arg176_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_66 = arg176_1 = None
        add_219 = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_99 = torch.ops.aten.sqrt.default(add_219);  add_219 = None
        reciprocal_99 = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_297 = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99 = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_793);  convolution_99 = unsqueeze_793 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_797);  mul_298 = unsqueeze_797 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_220 = torch.ops.aten.add.Tensor(mul_299, unsqueeze_799);  mul_299 = unsqueeze_799 = None
        add_221 = torch.ops.aten.add.Tensor(add_220, add_214);  add_220 = add_214 = None
        convolution_100 = torch.ops.aten.convolution.default(add_221, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_221 = arg181_1 = None
        add_222 = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_100 = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_100 = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_300 = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100 = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_801);  convolution_100 = unsqueeze_801 = None
        mul_301 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_805);  mul_301 = unsqueeze_805 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_223 = torch.ops.aten.add.Tensor(mul_302, unsqueeze_807);  mul_302 = unsqueeze_807 = None
        relu_67 = torch.ops.aten.relu.default(add_223);  add_223 = None
        convolution_101 = torch.ops.aten.convolution.default(relu_67, arg186_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 480);  relu_67 = arg186_1 = None
        add_224 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_101 = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_101 = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_303 = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101 = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_809);  convolution_101 = unsqueeze_809 = None
        mul_304 = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_813);  mul_304 = unsqueeze_813 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_225 = torch.ops.aten.add.Tensor(mul_305, unsqueeze_815);  mul_305 = unsqueeze_815 = None
        relu_68 = torch.ops.aten.relu.default(add_225);  add_225 = None
        convolution_102 = torch.ops.aten.convolution.default(relu_68, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_68 = arg191_1 = None
        add_226 = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_102 = torch.ops.aten.sqrt.default(add_226);  add_226 = None
        reciprocal_102 = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_306 = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102 = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_817);  convolution_102 = unsqueeze_817 = None
        mul_307 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_821);  mul_307 = unsqueeze_821 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_227 = torch.ops.aten.add.Tensor(mul_308, unsqueeze_823);  mul_308 = unsqueeze_823 = None
        convolution_103 = torch.ops.aten.convolution.default(add_227, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg196_1 = None
        add_228 = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_103 = torch.ops.aten.sqrt.default(add_228);  add_228 = None
        reciprocal_103 = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_309 = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103 = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_825);  convolution_103 = unsqueeze_825 = None
        mul_310 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_829);  mul_310 = unsqueeze_829 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_229 = torch.ops.aten.add.Tensor(mul_311, unsqueeze_831);  mul_311 = unsqueeze_831 = None
        relu_69 = torch.ops.aten.relu.default(add_229);  add_229 = None
        convolution_104 = torch.ops.aten.convolution.default(relu_69, arg201_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 288);  relu_69 = arg201_1 = None
        add_230 = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_104 = torch.ops.aten.sqrt.default(add_230);  add_230 = None
        reciprocal_104 = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_312 = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_832 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_833 = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        unsqueeze_834 = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
        unsqueeze_835 = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        sub_104 = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_833);  convolution_104 = unsqueeze_833 = None
        mul_313 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836 = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_837 = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_837);  mul_313 = unsqueeze_837 = None
        unsqueeze_838 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_839 = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_231 = torch.ops.aten.add.Tensor(mul_314, unsqueeze_839);  mul_314 = unsqueeze_839 = None
        relu_70 = torch.ops.aten.relu.default(add_231);  add_231 = None
        convolution_105 = torch.ops.aten.convolution.default(relu_70, arg206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_70 = arg206_1 = None
        add_232 = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_105 = torch.ops.aten.sqrt.default(add_232);  add_232 = None
        reciprocal_105 = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_315 = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_840 = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_841 = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        unsqueeze_842 = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
        unsqueeze_843 = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        sub_105 = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_841);  convolution_105 = unsqueeze_841 = None
        mul_316 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_845 = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_317 = torch.ops.aten.mul.Tensor(mul_316, unsqueeze_845);  mul_316 = unsqueeze_845 = None
        unsqueeze_846 = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_847 = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_233 = torch.ops.aten.add.Tensor(mul_317, unsqueeze_847);  mul_317 = unsqueeze_847 = None
        add_234 = torch.ops.aten.add.Tensor(add_233, add_227);  add_233 = add_227 = None
        convolution_106 = torch.ops.aten.convolution.default(add_234, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg211_1 = None
        add_235 = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_106 = torch.ops.aten.sqrt.default(add_235);  add_235 = None
        reciprocal_106 = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_318 = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_848 = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_849 = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        unsqueeze_850 = torch.ops.aten.unsqueeze.default(mul_318, -1);  mul_318 = None
        unsqueeze_851 = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        sub_106 = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_849);  convolution_106 = unsqueeze_849 = None
        mul_319 = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_853 = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, unsqueeze_853);  mul_319 = unsqueeze_853 = None
        unsqueeze_854 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_855 = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_236 = torch.ops.aten.add.Tensor(mul_320, unsqueeze_855);  mul_320 = unsqueeze_855 = None
        relu_71 = torch.ops.aten.relu.default(add_236);  add_236 = None
        convolution_107 = torch.ops.aten.convolution.default(relu_71, arg216_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 288);  relu_71 = arg216_1 = None
        add_237 = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_107 = torch.ops.aten.sqrt.default(add_237);  add_237 = None
        reciprocal_107 = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_321 = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_856 = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_857 = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        unsqueeze_858 = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
        unsqueeze_859 = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        sub_107 = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_857);  convolution_107 = unsqueeze_857 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_861 = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_861);  mul_322 = unsqueeze_861 = None
        unsqueeze_862 = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_863 = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_238 = torch.ops.aten.add.Tensor(mul_323, unsqueeze_863);  mul_323 = unsqueeze_863 = None
        relu_72 = torch.ops.aten.relu.default(add_238);  add_238 = None
        convolution_108 = torch.ops.aten.convolution.default(relu_72, arg221_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_72 = arg221_1 = None
        add_239 = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_108 = torch.ops.aten.sqrt.default(add_239);  add_239 = None
        reciprocal_108 = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_324 = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_864 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_865 = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        unsqueeze_866 = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
        unsqueeze_867 = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        sub_108 = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_865);  convolution_108 = unsqueeze_865 = None
        mul_325 = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868 = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_869 = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_869);  mul_325 = unsqueeze_869 = None
        unsqueeze_870 = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_871 = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_240 = torch.ops.aten.add.Tensor(mul_326, unsqueeze_871);  mul_326 = unsqueeze_871 = None
        add_241 = torch.ops.aten.add.Tensor(add_240, add_234);  add_240 = add_234 = None
        convolution_109 = torch.ops.aten.convolution.default(add_241, arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg226_1 = None
        add_242 = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_109 = torch.ops.aten.sqrt.default(add_242);  add_242 = None
        reciprocal_109 = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_327 = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_872 = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_873 = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        unsqueeze_874 = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_875 = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        sub_109 = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_873);  convolution_109 = unsqueeze_873 = None
        mul_328 = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876 = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_877 = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_877);  mul_328 = unsqueeze_877 = None
        unsqueeze_878 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_879 = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_243 = torch.ops.aten.add.Tensor(mul_329, unsqueeze_879);  mul_329 = unsqueeze_879 = None
        relu_73 = torch.ops.aten.relu.default(add_243);  add_243 = None
        convolution_110 = torch.ops.aten.convolution.default(relu_73, arg231_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 288);  relu_73 = arg231_1 = None
        add_244 = torch.ops.aten.add.Tensor(arg233_1, 1e-05);  arg233_1 = None
        sqrt_110 = torch.ops.aten.sqrt.default(add_244);  add_244 = None
        reciprocal_110 = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_330 = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_880 = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_881 = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        unsqueeze_882 = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_883 = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        sub_110 = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_881);  convolution_110 = unsqueeze_881 = None
        mul_331 = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884 = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_885 = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_885);  mul_331 = unsqueeze_885 = None
        unsqueeze_886 = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_887 = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_245 = torch.ops.aten.add.Tensor(mul_332, unsqueeze_887);  mul_332 = unsqueeze_887 = None
        relu_74 = torch.ops.aten.relu.default(add_245);  add_245 = None
        convolution_111 = torch.ops.aten.convolution.default(relu_74, arg236_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_74 = arg236_1 = None
        add_246 = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_111 = torch.ops.aten.sqrt.default(add_246);  add_246 = None
        reciprocal_111 = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_333 = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_888 = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_889 = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        unsqueeze_890 = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_891 = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        sub_111 = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_889);  convolution_111 = unsqueeze_889 = None
        mul_334 = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_893 = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_893);  mul_334 = unsqueeze_893 = None
        unsqueeze_894 = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_895 = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_247 = torch.ops.aten.add.Tensor(mul_335, unsqueeze_895);  mul_335 = unsqueeze_895 = None
        add_248 = torch.ops.aten.add.Tensor(add_247, add_241);  add_247 = add_241 = None
        convolution_112 = torch.ops.aten.convolution.default(add_248, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_248 = arg241_1 = None
        add_249 = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_112 = torch.ops.aten.sqrt.default(add_249);  add_249 = None
        reciprocal_112 = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_336 = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_896 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_897 = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        unsqueeze_898 = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_899 = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        sub_112 = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_897);  convolution_112 = unsqueeze_897 = None
        mul_337 = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_901 = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_901);  mul_337 = unsqueeze_901 = None
        unsqueeze_902 = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_903 = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_250 = torch.ops.aten.add.Tensor(mul_338, unsqueeze_903);  mul_338 = unsqueeze_903 = None
        relu_75 = torch.ops.aten.relu.default(add_250);  add_250 = None
        convolution_113 = torch.ops.aten.convolution.default(relu_75, arg246_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 576);  relu_75 = arg246_1 = None
        add_251 = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_113 = torch.ops.aten.sqrt.default(add_251);  add_251 = None
        reciprocal_113 = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_339 = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_904 = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_905 = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        unsqueeze_906 = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_907 = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        sub_113 = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_905);  convolution_113 = unsqueeze_905 = None
        mul_340 = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908 = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_909 = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_909);  mul_340 = unsqueeze_909 = None
        unsqueeze_910 = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_911 = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_252 = torch.ops.aten.add.Tensor(mul_341, unsqueeze_911);  mul_341 = unsqueeze_911 = None
        relu_76 = torch.ops.aten.relu.default(add_252);  add_252 = None
        convolution_114 = torch.ops.aten.convolution.default(relu_76, arg251_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_76 = arg251_1 = None
        add_253 = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_114 = torch.ops.aten.sqrt.default(add_253);  add_253 = None
        reciprocal_114 = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_342 = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_912 = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_913 = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        unsqueeze_914 = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_915 = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        sub_114 = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_913);  convolution_114 = unsqueeze_913 = None
        mul_343 = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_917 = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_917);  mul_343 = unsqueeze_917 = None
        unsqueeze_918 = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_919 = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_254 = torch.ops.aten.add.Tensor(mul_344, unsqueeze_919);  mul_344 = unsqueeze_919 = None
        convolution_115 = torch.ops.aten.convolution.default(add_254, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg256_1 = None
        add_255 = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_115 = torch.ops.aten.sqrt.default(add_255);  add_255 = None
        reciprocal_115 = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_345 = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_920 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_921 = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        unsqueeze_922 = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
        unsqueeze_923 = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        sub_115 = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_921);  convolution_115 = unsqueeze_921 = None
        mul_346 = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924 = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_925 = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_925);  mul_346 = unsqueeze_925 = None
        unsqueeze_926 = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_927 = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_256 = torch.ops.aten.add.Tensor(mul_347, unsqueeze_927);  mul_347 = unsqueeze_927 = None
        relu_77 = torch.ops.aten.relu.default(add_256);  add_256 = None
        convolution_116 = torch.ops.aten.convolution.default(relu_77, arg261_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  relu_77 = arg261_1 = None
        add_257 = torch.ops.aten.add.Tensor(arg263_1, 1e-05);  arg263_1 = None
        sqrt_116 = torch.ops.aten.sqrt.default(add_257);  add_257 = None
        reciprocal_116 = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_348 = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_928 = torch.ops.aten.unsqueeze.default(arg262_1, -1);  arg262_1 = None
        unsqueeze_929 = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        unsqueeze_930 = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_931 = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        sub_116 = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_929);  convolution_116 = unsqueeze_929 = None
        mul_349 = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932 = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_933 = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_933);  mul_349 = unsqueeze_933 = None
        unsqueeze_934 = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_935 = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_258 = torch.ops.aten.add.Tensor(mul_350, unsqueeze_935);  mul_350 = unsqueeze_935 = None
        relu_78 = torch.ops.aten.relu.default(add_258);  add_258 = None
        convolution_117 = torch.ops.aten.convolution.default(relu_78, arg266_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_78 = arg266_1 = None
        add_259 = torch.ops.aten.add.Tensor(arg268_1, 1e-05);  arg268_1 = None
        sqrt_117 = torch.ops.aten.sqrt.default(add_259);  add_259 = None
        reciprocal_117 = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_351 = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_936 = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_937 = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        unsqueeze_938 = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_939 = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        sub_117 = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_937);  convolution_117 = unsqueeze_937 = None
        mul_352 = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940 = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_941 = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_941);  mul_352 = unsqueeze_941 = None
        unsqueeze_942 = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_943 = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_260 = torch.ops.aten.add.Tensor(mul_353, unsqueeze_943);  mul_353 = unsqueeze_943 = None
        add_261 = torch.ops.aten.add.Tensor(add_260, add_254);  add_260 = add_254 = None
        convolution_118 = torch.ops.aten.convolution.default(add_261, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg271_1 = None
        add_262 = torch.ops.aten.add.Tensor(arg273_1, 1e-05);  arg273_1 = None
        sqrt_118 = torch.ops.aten.sqrt.default(add_262);  add_262 = None
        reciprocal_118 = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_354 = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_944 = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_945 = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        unsqueeze_946 = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
        unsqueeze_947 = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        sub_118 = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_945);  convolution_118 = unsqueeze_945 = None
        mul_355 = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948 = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_949 = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_949);  mul_355 = unsqueeze_949 = None
        unsqueeze_950 = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_951 = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_263 = torch.ops.aten.add.Tensor(mul_356, unsqueeze_951);  mul_356 = unsqueeze_951 = None
        relu_79 = torch.ops.aten.relu.default(add_263);  add_263 = None
        convolution_119 = torch.ops.aten.convolution.default(relu_79, arg276_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  relu_79 = arg276_1 = None
        add_264 = torch.ops.aten.add.Tensor(arg278_1, 1e-05);  arg278_1 = None
        sqrt_119 = torch.ops.aten.sqrt.default(add_264);  add_264 = None
        reciprocal_119 = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_357 = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_952 = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_953 = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        unsqueeze_954 = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
        unsqueeze_955 = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        sub_119 = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_953);  convolution_119 = unsqueeze_953 = None
        mul_358 = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956 = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_957 = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_957);  mul_358 = unsqueeze_957 = None
        unsqueeze_958 = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_959 = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_265 = torch.ops.aten.add.Tensor(mul_359, unsqueeze_959);  mul_359 = unsqueeze_959 = None
        relu_80 = torch.ops.aten.relu.default(add_265);  add_265 = None
        convolution_120 = torch.ops.aten.convolution.default(relu_80, arg281_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_80 = arg281_1 = None
        add_266 = torch.ops.aten.add.Tensor(arg283_1, 1e-05);  arg283_1 = None
        sqrt_120 = torch.ops.aten.sqrt.default(add_266);  add_266 = None
        reciprocal_120 = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_360 = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_960 = torch.ops.aten.unsqueeze.default(arg282_1, -1);  arg282_1 = None
        unsqueeze_961 = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        unsqueeze_962 = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_963 = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        sub_120 = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_961);  convolution_120 = unsqueeze_961 = None
        mul_361 = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964 = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_965 = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_965);  mul_361 = unsqueeze_965 = None
        unsqueeze_966 = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_967 = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_267 = torch.ops.aten.add.Tensor(mul_362, unsqueeze_967);  mul_362 = unsqueeze_967 = None
        add_268 = torch.ops.aten.add.Tensor(add_267, add_261);  add_267 = add_261 = None
        convolution_121 = torch.ops.aten.convolution.default(add_268, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg286_1 = None
        add_269 = torch.ops.aten.add.Tensor(arg288_1, 1e-05);  arg288_1 = None
        sqrt_121 = torch.ops.aten.sqrt.default(add_269);  add_269 = None
        reciprocal_121 = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_363 = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_968 = torch.ops.aten.unsqueeze.default(arg287_1, -1);  arg287_1 = None
        unsqueeze_969 = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        unsqueeze_970 = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
        unsqueeze_971 = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        sub_121 = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_969);  convolution_121 = unsqueeze_969 = None
        mul_364 = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972 = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_973 = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_973);  mul_364 = unsqueeze_973 = None
        unsqueeze_974 = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_975 = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_270 = torch.ops.aten.add.Tensor(mul_365, unsqueeze_975);  mul_365 = unsqueeze_975 = None
        relu_81 = torch.ops.aten.relu.default(add_270);  add_270 = None
        convolution_122 = torch.ops.aten.convolution.default(relu_81, arg291_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1152);  relu_81 = arg291_1 = None
        add_271 = torch.ops.aten.add.Tensor(arg293_1, 1e-05);  arg293_1 = None
        sqrt_122 = torch.ops.aten.sqrt.default(add_271);  add_271 = None
        reciprocal_122 = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_366 = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_976 = torch.ops.aten.unsqueeze.default(arg292_1, -1);  arg292_1 = None
        unsqueeze_977 = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        unsqueeze_978 = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_979 = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        sub_122 = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_977);  convolution_122 = unsqueeze_977 = None
        mul_367 = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980 = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_981 = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_981);  mul_367 = unsqueeze_981 = None
        unsqueeze_982 = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_983 = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_272 = torch.ops.aten.add.Tensor(mul_368, unsqueeze_983);  mul_368 = unsqueeze_983 = None
        relu_82 = torch.ops.aten.relu.default(add_272);  add_272 = None
        convolution_123 = torch.ops.aten.convolution.default(relu_82, arg296_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_82 = arg296_1 = None
        add_273 = torch.ops.aten.add.Tensor(arg298_1, 1e-05);  arg298_1 = None
        sqrt_123 = torch.ops.aten.sqrt.default(add_273);  add_273 = None
        reciprocal_123 = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_369 = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_984 = torch.ops.aten.unsqueeze.default(arg297_1, -1);  arg297_1 = None
        unsqueeze_985 = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        unsqueeze_986 = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_987 = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        sub_123 = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_985);  convolution_123 = unsqueeze_985 = None
        mul_370 = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988 = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_989 = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_989);  mul_370 = unsqueeze_989 = None
        unsqueeze_990 = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_991 = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_274 = torch.ops.aten.add.Tensor(mul_371, unsqueeze_991);  mul_371 = unsqueeze_991 = None
        add_275 = torch.ops.aten.add.Tensor(add_274, add_268);  add_274 = add_268 = None
        convolution_124 = torch.ops.aten.convolution.default(add_275, arg301_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_275 = arg301_1 = None
        add_276 = torch.ops.aten.add.Tensor(arg303_1, 1e-05);  arg303_1 = None
        sqrt_124 = torch.ops.aten.sqrt.default(add_276);  add_276 = None
        reciprocal_124 = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_372 = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_992 = torch.ops.aten.unsqueeze.default(arg302_1, -1);  arg302_1 = None
        unsqueeze_993 = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        unsqueeze_994 = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_995 = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        sub_124 = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_993);  convolution_124 = unsqueeze_993 = None
        mul_373 = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996 = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_997 = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_997);  mul_373 = unsqueeze_997 = None
        unsqueeze_998 = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_999 = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_277 = torch.ops.aten.add.Tensor(mul_374, unsqueeze_999);  mul_374 = unsqueeze_999 = None
        relu_83 = torch.ops.aten.relu.default(add_277);  add_277 = None
        convolution_125 = torch.ops.aten.convolution.default(relu_83, arg306_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1152);  relu_83 = arg306_1 = None
        add_278 = torch.ops.aten.add.Tensor(arg308_1, 1e-05);  arg308_1 = None
        sqrt_125 = torch.ops.aten.sqrt.default(add_278);  add_278 = None
        reciprocal_125 = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_375 = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1000 = torch.ops.aten.unsqueeze.default(arg307_1, -1);  arg307_1 = None
        unsqueeze_1001 = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        unsqueeze_1002 = torch.ops.aten.unsqueeze.default(mul_375, -1);  mul_375 = None
        unsqueeze_1003 = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        sub_125 = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_1001);  convolution_125 = unsqueeze_1001 = None
        mul_376 = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004 = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_1005 = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_1005);  mul_376 = unsqueeze_1005 = None
        unsqueeze_1006 = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_1007 = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_279 = torch.ops.aten.add.Tensor(mul_377, unsqueeze_1007);  mul_377 = unsqueeze_1007 = None
        relu_84 = torch.ops.aten.relu.default(add_279);  add_279 = None
        convolution_126 = torch.ops.aten.convolution.default(relu_84, arg311_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_84 = arg311_1 = None
        add_280 = torch.ops.aten.add.Tensor(arg313_1, 1e-05);  arg313_1 = None
        sqrt_126 = torch.ops.aten.sqrt.default(add_280);  add_280 = None
        reciprocal_126 = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_378 = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1008 = torch.ops.aten.unsqueeze.default(arg312_1, -1);  arg312_1 = None
        unsqueeze_1009 = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        unsqueeze_1010 = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_1011 = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        sub_126 = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1009);  convolution_126 = unsqueeze_1009 = None
        mul_379 = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012 = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_1013 = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_1013);  mul_379 = unsqueeze_1013 = None
        unsqueeze_1014 = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_1015 = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_281 = torch.ops.aten.add.Tensor(mul_380, unsqueeze_1015);  mul_380 = unsqueeze_1015 = None
        convolution_127 = torch.ops.aten.convolution.default(add_281, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_281 = arg316_1 = None
        add_282 = torch.ops.aten.add.Tensor(arg318_1, 1e-05);  arg318_1 = None
        sqrt_127 = torch.ops.aten.sqrt.default(add_282);  add_282 = None
        reciprocal_127 = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_381 = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1016 = torch.ops.aten.unsqueeze.default(arg317_1, -1);  arg317_1 = None
        unsqueeze_1017 = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        unsqueeze_1018 = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_1019 = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        sub_127 = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_1017);  convolution_127 = unsqueeze_1017 = None
        mul_382 = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020 = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_1021 = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_1021);  mul_382 = unsqueeze_1021 = None
        unsqueeze_1022 = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_1023 = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_283 = torch.ops.aten.add.Tensor(mul_383, unsqueeze_1023);  mul_383 = unsqueeze_1023 = None
        relu_85 = torch.ops.aten.relu.default(add_283);  add_283 = None
        mean_1 = torch.ops.aten.mean.dim(relu_85, [-1, -2], True);  relu_85 = None
        view_1 = torch.ops.aten.view.default(mean_1, [8, 1280]);  mean_1 = None
        permute_1 = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg322_1, view_1, permute_1);  arg322_1 = view_1 = permute_1 = None
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
    buf16 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf16, (48, 16, 1, 1), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf17, (48,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf18, (48,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf19, (48,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf20, (48,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf21, (48, 1, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf22, (48,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf23, (48,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf24, (48,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 192, device=device(type='cuda', index=0))
    reader.tensor(buf25, (48,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf26, (24, 48, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf27, (24,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf28, (24,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf29, (24,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf30, (24,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf31, (72, 24, 1, 1), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf32, (72,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf33, (72,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf34, (72,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf35, (72,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf36, (72, 1, 3, 3), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf37, (72,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf38, (72,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf39, (72,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf40, (72,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf41, (24, 72, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf42, (24,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf43, (24,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf44, (24,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf45, (24,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf46, (72, 24, 1, 1), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf47, (72,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf48, (72,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf49, (72,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf50, (72,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf51, (72, 1, 3, 3), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf52, (72,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf53, (72,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf54, (72,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf55, (72,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf56, (24, 72, 1, 1), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf57, (24,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf58, (24,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf59, (24,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf60, (24,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf61, (144, 24, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf62, (144,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf63, (144,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf64, (144,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf65, (144,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 14400, device=device(type='cuda', index=0))
    reader.tensor(buf66, (144, 1, 5, 5), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf67, (144,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf68, (144,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf69, (144,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf70, (144,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 23040, device=device(type='cuda', index=0))
    reader.tensor(buf71, (40, 144, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf72, (40,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf73, (40,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf74, (40,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf75, (40,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf76, (120, 40, 1, 1), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf77, (120,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf78, (120,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf79, (120,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf80, (120,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 4320, device=device(type='cuda', index=0))
    reader.tensor(buf81, (120, 1, 3, 3), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf82, (120,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf83, (120,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf84, (120,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf85, (120,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf86, (40, 120, 1, 1), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf87, (40,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf88, (40,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf89, (40,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf90, (40,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf91, (120, 40, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf92, (120,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf93, (120,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf94, (120,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf95, (120,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 4320, device=device(type='cuda', index=0))
    reader.tensor(buf96, (120, 1, 3, 3), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf97, (120,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf98, (120,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf99, (120,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf100, (120,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf101, (40, 120, 1, 1), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf102, (40,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf103, (40,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf104, (40,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf105, (40,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf106, (120, 40, 1, 1), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf107, (120,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf108, (120,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf109, (120,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf110, (120,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 4320, device=device(type='cuda', index=0))
    reader.tensor(buf111, (120, 1, 3, 3), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf112, (120,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf113, (120,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf114, (120,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 480, device=device(type='cuda', index=0))
    reader.tensor(buf115, (120,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 19200, device=device(type='cuda', index=0))
    reader.tensor(buf116, (40, 120, 1, 1), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf117, (40,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf118, (40,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf119, (40,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 160, device=device(type='cuda', index=0))
    reader.tensor(buf120, (40,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 38400, device=device(type='cuda', index=0))
    reader.tensor(buf121, (240, 40, 1, 1), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf122, (240,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf123, (240,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf124, (240,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf125, (240,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 24000, device=device(type='cuda', index=0))
    reader.tensor(buf126, (240, 1, 5, 5), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf127, (240,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf128, (240,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf129, (240,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf130, (240,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf131, (80, 240, 1, 1), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf132, (80,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf133, (80,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf134, (80,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf135, (80,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf136, (240, 80, 1, 1), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf137, (240,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf138, (240,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf139, (240,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf140, (240,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 8640, device=device(type='cuda', index=0))
    reader.tensor(buf141, (240, 1, 3, 3), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf142, (240,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf143, (240,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf144, (240,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf145, (240,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf146, (80, 240, 1, 1), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf147, (80,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf148, (80,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf149, (80,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf150, (80,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf151, (240, 80, 1, 1), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf152, (240,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf153, (240,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf154, (240,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf155, (240,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 8640, device=device(type='cuda', index=0))
    reader.tensor(buf156, (240, 1, 3, 3), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf157, (240,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf158, (240,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf159, (240,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf160, (240,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf161, (80, 240, 1, 1), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf162, (80,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf163, (80,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf164, (80,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf165, (80,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf166, (240, 80, 1, 1), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf167, (240,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf168, (240,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf169, (240,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf170, (240,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 8640, device=device(type='cuda', index=0))
    reader.tensor(buf171, (240, 1, 3, 3), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf172, (240,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf173, (240,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf174, (240,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf175, (240,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 76800, device=device(type='cuda', index=0))
    reader.tensor(buf176, (80, 240, 1, 1), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf177, (80,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf178, (80,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf179, (80,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 320, device=device(type='cuda', index=0))
    reader.tensor(buf180, (80,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf181, (480, 80, 1, 1), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf182, (480,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf183, (480,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf184, (480,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf185, (480,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 48000, device=device(type='cuda', index=0))
    reader.tensor(buf186, (480, 1, 5, 5), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf187, (480,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf188, (480,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf189, (480,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf190, (480,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 184320, device=device(type='cuda', index=0))
    reader.tensor(buf191, (96, 480, 1, 1), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf192, (96,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf193, (96,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf194, (96,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf195, (96,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 110592, device=device(type='cuda', index=0))
    reader.tensor(buf196, (288, 96, 1, 1), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf197, (288,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf198, (288,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf199, (288,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf200, (288,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 28800, device=device(type='cuda', index=0))
    reader.tensor(buf201, (288, 1, 5, 5), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf202, (288,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf203, (288,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf204, (288,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf205, (288,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 110592, device=device(type='cuda', index=0))
    reader.tensor(buf206, (96, 288, 1, 1), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf207, (96,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf208, (96,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf209, (96,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf210, (96,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 110592, device=device(type='cuda', index=0))
    reader.tensor(buf211, (288, 96, 1, 1), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf212, (288,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf213, (288,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf214, (288,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf215, (288,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 28800, device=device(type='cuda', index=0))
    reader.tensor(buf216, (288, 1, 5, 5), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf217, (288,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf218, (288,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf219, (288,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf220, (288,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 110592, device=device(type='cuda', index=0))
    reader.tensor(buf221, (96, 288, 1, 1), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf222, (96,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf223, (96,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf224, (96,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf225, (96,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 110592, device=device(type='cuda', index=0))
    reader.tensor(buf226, (288, 96, 1, 1), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf227, (288,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf228, (288,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf229, (288,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf230, (288,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 28800, device=device(type='cuda', index=0))
    reader.tensor(buf231, (288, 1, 5, 5), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf232, (288,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf233, (288,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf234, (288,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf235, (288,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 110592, device=device(type='cuda', index=0))
    reader.tensor(buf236, (96, 288, 1, 1), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf237, (96,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf238, (96,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf239, (96,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf240, (96,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf241, (576, 96, 1, 1), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf242, (576,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf243, (576,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf244, (576,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf245, (576,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 57600, device=device(type='cuda', index=0))
    reader.tensor(buf246, (576, 1, 5, 5), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf247, (576,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf248, (576,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf249, (576,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf250, (576,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf251, (192, 576, 1, 1), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf252, (192,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf253, (192,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf254, (192,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf255, (192,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf256, (1152, 192, 1, 1), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf257, (1152,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf258, (1152,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf259, (1152,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1152,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1152, 1, 5, 5), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf262, (1152,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf263, (1152,), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf264, (1152,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf265, (1152,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf266, (192, 1152, 1, 1), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf267, (192,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf268, (192,), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf269, (192,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf270, (192,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf271, (1152, 192, 1, 1), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf272, (1152,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf273, (1152,), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf274, (1152,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf275, (1152,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 115200, device=device(type='cuda', index=0))
    reader.tensor(buf276, (1152, 1, 5, 5), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf277, (1152,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf278, (1152,), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf279, (1152,), is_leaf=True)  # arg279_1
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
    buf296 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf296, (192, 1152, 1, 1), is_leaf=True)  # arg296_1
    buf297 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf297, (192,), is_leaf=True)  # arg297_1
    buf298 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf298, (192,), is_leaf=True)  # arg298_1
    buf299 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf299, (192,), is_leaf=True)  # arg299_1
    buf300 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf300, (192,), is_leaf=True)  # arg300_1
    buf301 = reader.storage(None, 884736, device=device(type='cuda', index=0))
    reader.tensor(buf301, (1152, 192, 1, 1), is_leaf=True)  # arg301_1
    buf302 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf302, (1152,), is_leaf=True)  # arg302_1
    buf303 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf303, (1152,), is_leaf=True)  # arg303_1
    buf304 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf304, (1152,), is_leaf=True)  # arg304_1
    buf305 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf305, (1152,), is_leaf=True)  # arg305_1
    buf306 = reader.storage(None, 41472, device=device(type='cuda', index=0))
    reader.tensor(buf306, (1152, 1, 3, 3), is_leaf=True)  # arg306_1
    buf307 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf307, (1152,), is_leaf=True)  # arg307_1
    buf308 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf308, (1152,), is_leaf=True)  # arg308_1
    buf309 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf309, (1152,), is_leaf=True)  # arg309_1
    buf310 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf310, (1152,), is_leaf=True)  # arg310_1
    buf311 = reader.storage(None, 1474560, device=device(type='cuda', index=0))
    reader.tensor(buf311, (320, 1152, 1, 1), is_leaf=True)  # arg311_1
    buf312 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf312, (320,), is_leaf=True)  # arg312_1
    buf313 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf313, (320,), is_leaf=True)  # arg313_1
    buf314 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf314, (320,), is_leaf=True)  # arg314_1
    buf315 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf315, (320,), is_leaf=True)  # arg315_1
    buf316 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf316, (1280, 320, 1, 1), is_leaf=True)  # arg316_1
    buf317 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf317, (1280,), is_leaf=True)  # arg317_1
    buf318 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf318, (1280,), is_leaf=True)  # arg318_1
    buf319 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf319, (1280,), is_leaf=True)  # arg319_1
    buf320 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf320, (1280,), is_leaf=True)  # arg320_1
    buf321 = reader.storage(None, 5120000, device=device(type='cuda', index=0))
    reader.tensor(buf321, (1000, 1280), is_leaf=True)  # arg321_1
    buf322 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf322, (1000,), is_leaf=True)  # arg322_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)