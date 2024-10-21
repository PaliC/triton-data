
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1):
        convolution_52 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_114 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_52 = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_156 = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
        mul_157 = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_115 = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
        clamp_min_35 = torch.ops.aten.clamp_min.default(add_115, 0.0);  add_115 = None
        clamp_max_35 = torch.ops.aten.clamp_max.default(clamp_min_35, 6.0);  clamp_min_35 = None
        convolution_53 = torch.ops.aten.convolution.default(clamp_max_35, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  clamp_max_35 = arg6_1 = None
        add_116 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_53 = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_53 = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_159 = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_53 = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_425);  convolution_53 = unsqueeze_425 = None
        mul_160 = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_117 = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
        clamp_min_36 = torch.ops.aten.clamp_min.default(add_117, 0.0);  add_117 = None
        clamp_max_36 = torch.ops.aten.clamp_max.default(clamp_min_36, 6.0);  clamp_min_36 = None
        convolution_54 = torch.ops.aten.convolution.default(clamp_max_36, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_36 = arg11_1 = None
        add_118 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_54 = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_54 = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_162 = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_54 = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  convolution_54 = unsqueeze_433 = None
        mul_163 = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_119 = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
        convolution_55 = torch.ops.aten.convolution.default(add_119, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_119 = arg16_1 = None
        add_120 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_55 = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_55 = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_165 = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_55 = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  convolution_55 = unsqueeze_441 = None
        mul_166 = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_121 = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
        clamp_min_37 = torch.ops.aten.clamp_min.default(add_121, 0.0);  add_121 = None
        clamp_max_37 = torch.ops.aten.clamp_max.default(clamp_min_37, 6.0);  clamp_min_37 = None
        convolution_56 = torch.ops.aten.convolution.default(clamp_max_37, arg21_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 96);  clamp_max_37 = arg21_1 = None
        add_122 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_56 = torch.ops.aten.sqrt.default(add_122);  add_122 = None
        reciprocal_56 = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_168 = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_56 = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_449);  convolution_56 = unsqueeze_449 = None
        mul_169 = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_170 = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_123 = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
        clamp_min_38 = torch.ops.aten.clamp_min.default(add_123, 0.0);  add_123 = None
        clamp_max_38 = torch.ops.aten.clamp_max.default(clamp_min_38, 6.0);  clamp_min_38 = None
        convolution_57 = torch.ops.aten.convolution.default(clamp_max_38, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_38 = arg26_1 = None
        add_124 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_57 = torch.ops.aten.sqrt.default(add_124);  add_124 = None
        reciprocal_57 = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_171 = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458 = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_459 = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_57 = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_457);  convolution_57 = unsqueeze_457 = None
        mul_172 = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
        unsqueeze_460 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_461 = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_125 = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
        convolution_58 = torch.ops.aten.convolution.default(add_125, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg31_1 = None
        add_126 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_58 = torch.ops.aten.sqrt.default(add_126);  add_126 = None
        reciprocal_58 = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_174 = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_58 = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_465);  convolution_58 = unsqueeze_465 = None
        mul_175 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_127 = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
        clamp_min_39 = torch.ops.aten.clamp_min.default(add_127, 0.0);  add_127 = None
        clamp_max_39 = torch.ops.aten.clamp_max.default(clamp_min_39, 6.0);  clamp_min_39 = None
        convolution_59 = torch.ops.aten.convolution.default(clamp_max_39, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 144);  clamp_max_39 = arg36_1 = None
        add_128 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_59 = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_59 = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_177 = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_59 = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_473);  convolution_59 = unsqueeze_473 = None
        mul_178 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_129 = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
        clamp_min_40 = torch.ops.aten.clamp_min.default(add_129, 0.0);  add_129 = None
        clamp_max_40 = torch.ops.aten.clamp_max.default(clamp_min_40, 6.0);  clamp_min_40 = None
        convolution_60 = torch.ops.aten.convolution.default(clamp_max_40, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_40 = arg41_1 = None
        add_130 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_60 = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_60 = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_180 = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_60 = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_481);  convolution_60 = unsqueeze_481 = None
        mul_181 = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_182 = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_131 = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
        add_132 = torch.ops.aten.add.Tensor(add_131, add_125);  add_131 = add_125 = None
        convolution_61 = torch.ops.aten.convolution.default(add_132, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_132 = arg46_1 = None
        add_133 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_61 = torch.ops.aten.sqrt.default(add_133);  add_133 = None
        reciprocal_61 = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_183 = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_61 = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_489);  convolution_61 = unsqueeze_489 = None
        mul_184 = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_134 = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
        clamp_min_41 = torch.ops.aten.clamp_min.default(add_134, 0.0);  add_134 = None
        clamp_max_41 = torch.ops.aten.clamp_max.default(clamp_min_41, 6.0);  clamp_min_41 = None
        convolution_62 = torch.ops.aten.convolution.default(clamp_max_41, arg51_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 144);  clamp_max_41 = arg51_1 = None
        add_135 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_62 = torch.ops.aten.sqrt.default(add_135);  add_135 = None
        reciprocal_62 = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_186 = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62 = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_497);  convolution_62 = unsqueeze_497 = None
        mul_187 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_136 = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
        clamp_min_42 = torch.ops.aten.clamp_min.default(add_136, 0.0);  add_136 = None
        clamp_max_42 = torch.ops.aten.clamp_max.default(clamp_min_42, 6.0);  clamp_min_42 = None
        convolution_63 = torch.ops.aten.convolution.default(clamp_max_42, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_42 = arg56_1 = None
        add_137 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_63 = torch.ops.aten.sqrt.default(add_137);  add_137 = None
        reciprocal_63 = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_189 = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63 = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_505);  convolution_63 = unsqueeze_505 = None
        mul_190 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_138 = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
        convolution_64 = torch.ops.aten.convolution.default(add_138, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg61_1 = None
        add_139 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_64 = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_64 = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_192 = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64 = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_513);  convolution_64 = unsqueeze_513 = None
        mul_193 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_140 = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
        clamp_min_43 = torch.ops.aten.clamp_min.default(add_140, 0.0);  add_140 = None
        clamp_max_43 = torch.ops.aten.clamp_max.default(clamp_min_43, 6.0);  clamp_min_43 = None
        convolution_65 = torch.ops.aten.convolution.default(clamp_max_43, arg66_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  clamp_max_43 = arg66_1 = None
        add_141 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_65 = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_65 = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_195 = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65 = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_521);  convolution_65 = unsqueeze_521 = None
        mul_196 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_142 = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
        clamp_min_44 = torch.ops.aten.clamp_min.default(add_142, 0.0);  add_142 = None
        clamp_max_44 = torch.ops.aten.clamp_max.default(clamp_min_44, 6.0);  clamp_min_44 = None
        convolution_66 = torch.ops.aten.convolution.default(clamp_max_44, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_44 = arg71_1 = None
        add_143 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_66 = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_66 = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_198 = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66 = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_529);  convolution_66 = unsqueeze_529 = None
        mul_199 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_144 = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
        add_145 = torch.ops.aten.add.Tensor(add_144, add_138);  add_144 = add_138 = None
        convolution_67 = torch.ops.aten.convolution.default(add_145, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg76_1 = None
        add_146 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_67 = torch.ops.aten.sqrt.default(add_146);  add_146 = None
        reciprocal_67 = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_201 = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_537);  convolution_67 = unsqueeze_537 = None
        mul_202 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_543 = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_147 = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
        clamp_min_45 = torch.ops.aten.clamp_min.default(add_147, 0.0);  add_147 = None
        clamp_max_45 = torch.ops.aten.clamp_max.default(clamp_min_45, 6.0);  clamp_min_45 = None
        convolution_68 = torch.ops.aten.convolution.default(clamp_max_45, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 192);  clamp_max_45 = arg81_1 = None
        add_148 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_68 = torch.ops.aten.sqrt.default(add_148);  add_148 = None
        reciprocal_68 = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_204 = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_545 = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68 = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_545);  convolution_68 = unsqueeze_545 = None
        mul_205 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_206 = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_149 = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
        clamp_min_46 = torch.ops.aten.clamp_min.default(add_149, 0.0);  add_149 = None
        clamp_max_46 = torch.ops.aten.clamp_max.default(clamp_min_46, 6.0);  clamp_min_46 = None
        convolution_69 = torch.ops.aten.convolution.default(clamp_max_46, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_46 = arg86_1 = None
        add_150 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_69 = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_69 = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_207 = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69 = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_553);  convolution_69 = unsqueeze_553 = None
        mul_208 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_151 = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
        add_152 = torch.ops.aten.add.Tensor(add_151, add_145);  add_151 = add_145 = None
        convolution_70 = torch.ops.aten.convolution.default(add_152, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_152 = arg91_1 = None
        add_153 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_70 = torch.ops.aten.sqrt.default(add_153);  add_153 = None
        reciprocal_70 = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_210 = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70 = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_561);  convolution_70 = unsqueeze_561 = None
        mul_211 = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_154 = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
        clamp_min_47 = torch.ops.aten.clamp_min.default(add_154, 0.0);  add_154 = None
        clamp_max_47 = torch.ops.aten.clamp_max.default(clamp_min_47, 6.0);  clamp_min_47 = None
        convolution_71 = torch.ops.aten.convolution.default(clamp_max_47, arg96_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 192);  clamp_max_47 = arg96_1 = None
        add_155 = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_71 = torch.ops.aten.sqrt.default(add_155);  add_155 = None
        reciprocal_71 = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_213 = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71 = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_569);  convolution_71 = unsqueeze_569 = None
        mul_214 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_215 = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_156 = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
        clamp_min_48 = torch.ops.aten.clamp_min.default(add_156, 0.0);  add_156 = None
        clamp_max_48 = torch.ops.aten.clamp_max.default(clamp_min_48, 6.0);  clamp_min_48 = None
        convolution_72 = torch.ops.aten.convolution.default(clamp_max_48, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_48 = arg101_1 = None
        add_157 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_72 = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_72 = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_216 = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72 = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_577);  convolution_72 = unsqueeze_577 = None
        mul_217 = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_218 = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_158 = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
        convolution_73 = torch.ops.aten.convolution.default(add_158, arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg106_1 = None
        add_159 = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_73 = torch.ops.aten.sqrt.default(add_159);  add_159 = None
        reciprocal_73 = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_219 = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73 = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_585);  convolution_73 = unsqueeze_585 = None
        mul_220 = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_160 = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
        clamp_min_49 = torch.ops.aten.clamp_min.default(add_160, 0.0);  add_160 = None
        clamp_max_49 = torch.ops.aten.clamp_max.default(clamp_min_49, 6.0);  clamp_min_49 = None
        convolution_74 = torch.ops.aten.convolution.default(clamp_max_49, arg111_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384);  clamp_max_49 = arg111_1 = None
        add_161 = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_74 = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_74 = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_222 = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74 = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_593);  convolution_74 = unsqueeze_593 = None
        mul_223 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_224 = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_162 = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
        clamp_min_50 = torch.ops.aten.clamp_min.default(add_162, 0.0);  add_162 = None
        clamp_max_50 = torch.ops.aten.clamp_max.default(clamp_min_50, 6.0);  clamp_min_50 = None
        convolution_75 = torch.ops.aten.convolution.default(clamp_max_50, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_50 = arg116_1 = None
        add_163 = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_75 = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_75 = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_225 = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75 = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_601);  convolution_75 = unsqueeze_601 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_605 = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_227 = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_164 = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
        add_165 = torch.ops.aten.add.Tensor(add_164, add_158);  add_164 = add_158 = None
        convolution_76 = torch.ops.aten.convolution.default(add_165, arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg121_1 = None
        add_166 = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_76 = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_76 = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_228 = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_611 = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76 = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_609);  convolution_76 = unsqueeze_609 = None
        mul_229 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_613 = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
        unsqueeze_614 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_615 = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_167 = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
        clamp_min_51 = torch.ops.aten.clamp_min.default(add_167, 0.0);  add_167 = None
        clamp_max_51 = torch.ops.aten.clamp_max.default(clamp_min_51, 6.0);  clamp_min_51 = None
        convolution_77 = torch.ops.aten.convolution.default(clamp_max_51, arg126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384);  clamp_max_51 = arg126_1 = None
        add_168 = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_77 = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_77 = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_231 = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_617 = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618 = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
        unsqueeze_619 = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77 = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_617);  convolution_77 = unsqueeze_617 = None
        mul_232 = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_623 = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_169 = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
        clamp_min_52 = torch.ops.aten.clamp_min.default(add_169, 0.0);  add_169 = None
        clamp_max_52 = torch.ops.aten.clamp_max.default(clamp_min_52, 6.0);  clamp_min_52 = None
        convolution_78 = torch.ops.aten.convolution.default(clamp_max_52, arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_52 = arg131_1 = None
        add_170 = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_78 = torch.ops.aten.sqrt.default(add_170);  add_170 = None
        reciprocal_78 = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_234 = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_625 = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626 = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
        unsqueeze_627 = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78 = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_625);  convolution_78 = unsqueeze_625 = None
        mul_235 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_629 = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
        unsqueeze_630 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_631 = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_171 = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
        add_172 = torch.ops.aten.add.Tensor(add_171, add_165);  add_171 = add_165 = None
        convolution_79 = torch.ops.aten.convolution.default(add_172, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg136_1 = None
        add_173 = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_79 = torch.ops.aten.sqrt.default(add_173);  add_173 = None
        reciprocal_79 = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_237 = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_635 = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79 = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_633);  convolution_79 = unsqueeze_633 = None
        mul_238 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_637 = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
        unsqueeze_638 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_639 = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_174 = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
        clamp_min_53 = torch.ops.aten.clamp_min.default(add_174, 0.0);  add_174 = None
        clamp_max_53 = torch.ops.aten.clamp_max.default(clamp_min_53, 6.0);  clamp_min_53 = None
        convolution_80 = torch.ops.aten.convolution.default(clamp_max_53, arg141_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384);  clamp_max_53 = arg141_1 = None
        add_175 = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_80 = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_80 = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_240 = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_641 = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80 = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_641);  convolution_80 = unsqueeze_641 = None
        mul_241 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_176 = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
        clamp_min_54 = torch.ops.aten.clamp_min.default(add_176, 0.0);  add_176 = None
        clamp_max_54 = torch.ops.aten.clamp_max.default(clamp_min_54, 6.0);  clamp_min_54 = None
        convolution_81 = torch.ops.aten.convolution.default(clamp_max_54, arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_54 = arg146_1 = None
        add_177 = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_81 = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_81 = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_243 = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_651 = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81 = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_649);  convolution_81 = unsqueeze_649 = None
        mul_244 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_653 = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_178 = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
        add_179 = torch.ops.aten.add.Tensor(add_178, add_172);  add_178 = add_172 = None
        convolution_82 = torch.ops.aten.convolution.default(add_179, arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_179 = arg151_1 = None
        add_180 = torch.ops.aten.add.Tensor(arg153_1, 1e-05);  arg153_1 = None
        sqrt_82 = torch.ops.aten.sqrt.default(add_180);  add_180 = None
        reciprocal_82 = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_246 = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_656 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_657 = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        unsqueeze_658 = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
        unsqueeze_659 = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        sub_82 = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_657);  convolution_82 = unsqueeze_657 = None
        mul_247 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_661 = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_661);  mul_247 = unsqueeze_661 = None
        unsqueeze_662 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_663 = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_181 = torch.ops.aten.add.Tensor(mul_248, unsqueeze_663);  mul_248 = unsqueeze_663 = None
        clamp_min_55 = torch.ops.aten.clamp_min.default(add_181, 0.0);  add_181 = None
        clamp_max_55 = torch.ops.aten.clamp_max.default(clamp_min_55, 6.0);  clamp_min_55 = None
        convolution_83 = torch.ops.aten.convolution.default(clamp_max_55, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 384);  clamp_max_55 = arg156_1 = None
        add_182 = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_83 = torch.ops.aten.sqrt.default(add_182);  add_182 = None
        reciprocal_83 = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_249 = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_664 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_665 = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        unsqueeze_666 = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
        unsqueeze_667 = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        sub_83 = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_665);  convolution_83 = unsqueeze_665 = None
        mul_250 = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668 = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_669 = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_669);  mul_250 = unsqueeze_669 = None
        unsqueeze_670 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_671 = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_183 = torch.ops.aten.add.Tensor(mul_251, unsqueeze_671);  mul_251 = unsqueeze_671 = None
        clamp_min_56 = torch.ops.aten.clamp_min.default(add_183, 0.0);  add_183 = None
        clamp_max_56 = torch.ops.aten.clamp_max.default(clamp_min_56, 6.0);  clamp_min_56 = None
        convolution_84 = torch.ops.aten.convolution.default(clamp_max_56, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_56 = arg161_1 = None
        add_184 = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_84 = torch.ops.aten.sqrt.default(add_184);  add_184 = None
        reciprocal_84 = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_252 = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_672 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_673 = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        unsqueeze_674 = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
        unsqueeze_675 = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        sub_84 = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_673);  convolution_84 = unsqueeze_673 = None
        mul_253 = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676 = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_677 = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_254 = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_677);  mul_253 = unsqueeze_677 = None
        unsqueeze_678 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_679 = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_185 = torch.ops.aten.add.Tensor(mul_254, unsqueeze_679);  mul_254 = unsqueeze_679 = None
        convolution_85 = torch.ops.aten.convolution.default(add_185, arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg166_1 = None
        add_186 = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_85 = torch.ops.aten.sqrt.default(add_186);  add_186 = None
        reciprocal_85 = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_255 = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_680 = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_681 = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        unsqueeze_682 = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_683 = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        sub_85 = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_681);  convolution_85 = unsqueeze_681 = None
        mul_256 = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_685 = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_685);  mul_256 = unsqueeze_685 = None
        unsqueeze_686 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_687 = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_187 = torch.ops.aten.add.Tensor(mul_257, unsqueeze_687);  mul_257 = unsqueeze_687 = None
        clamp_min_57 = torch.ops.aten.clamp_min.default(add_187, 0.0);  add_187 = None
        clamp_max_57 = torch.ops.aten.clamp_max.default(clamp_min_57, 6.0);  clamp_min_57 = None
        convolution_86 = torch.ops.aten.convolution.default(clamp_max_57, arg171_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 576);  clamp_max_57 = arg171_1 = None
        add_188 = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_86 = torch.ops.aten.sqrt.default(add_188);  add_188 = None
        reciprocal_86 = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_258 = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_688 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_689 = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        unsqueeze_690 = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
        unsqueeze_691 = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        sub_86 = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_689);  convolution_86 = unsqueeze_689 = None
        mul_259 = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692 = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_693 = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_260 = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_693);  mul_259 = unsqueeze_693 = None
        unsqueeze_694 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_695 = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_189 = torch.ops.aten.add.Tensor(mul_260, unsqueeze_695);  mul_260 = unsqueeze_695 = None
        clamp_min_58 = torch.ops.aten.clamp_min.default(add_189, 0.0);  add_189 = None
        clamp_max_58 = torch.ops.aten.clamp_max.default(clamp_min_58, 6.0);  clamp_min_58 = None
        convolution_87 = torch.ops.aten.convolution.default(clamp_max_58, arg176_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_58 = arg176_1 = None
        add_190 = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_87 = torch.ops.aten.sqrt.default(add_190);  add_190 = None
        reciprocal_87 = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_261 = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_696 = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_697 = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        unsqueeze_698 = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_699 = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        sub_87 = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_697);  convolution_87 = unsqueeze_697 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700 = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_701 = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_701);  mul_262 = unsqueeze_701 = None
        unsqueeze_702 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_703 = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_191 = torch.ops.aten.add.Tensor(mul_263, unsqueeze_703);  mul_263 = unsqueeze_703 = None
        add_192 = torch.ops.aten.add.Tensor(add_191, add_185);  add_191 = add_185 = None
        convolution_88 = torch.ops.aten.convolution.default(add_192, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg181_1 = None
        add_193 = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_88 = torch.ops.aten.sqrt.default(add_193);  add_193 = None
        reciprocal_88 = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_264 = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_704 = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_705 = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        unsqueeze_706 = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_707 = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        sub_88 = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_705);  convolution_88 = unsqueeze_705 = None
        mul_265 = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708 = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_709 = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_709);  mul_265 = unsqueeze_709 = None
        unsqueeze_710 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_711 = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_194 = torch.ops.aten.add.Tensor(mul_266, unsqueeze_711);  mul_266 = unsqueeze_711 = None
        clamp_min_59 = torch.ops.aten.clamp_min.default(add_194, 0.0);  add_194 = None
        clamp_max_59 = torch.ops.aten.clamp_max.default(clamp_min_59, 6.0);  clamp_min_59 = None
        convolution_89 = torch.ops.aten.convolution.default(clamp_max_59, arg186_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 576);  clamp_max_59 = arg186_1 = None
        add_195 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_89 = torch.ops.aten.sqrt.default(add_195);  add_195 = None
        reciprocal_89 = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_267 = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_712 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_713 = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        unsqueeze_714 = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_715 = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        sub_89 = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_713);  convolution_89 = unsqueeze_713 = None
        mul_268 = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_717 = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_717);  mul_268 = unsqueeze_717 = None
        unsqueeze_718 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_719 = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_196 = torch.ops.aten.add.Tensor(mul_269, unsqueeze_719);  mul_269 = unsqueeze_719 = None
        clamp_min_60 = torch.ops.aten.clamp_min.default(add_196, 0.0);  add_196 = None
        clamp_max_60 = torch.ops.aten.clamp_max.default(clamp_min_60, 6.0);  clamp_min_60 = None
        convolution_90 = torch.ops.aten.convolution.default(clamp_max_60, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_60 = arg191_1 = None
        add_197 = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_90 = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_90 = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_270 = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_720 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_721 = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        unsqueeze_722 = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_723 = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        sub_90 = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_721);  convolution_90 = unsqueeze_721 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_725 = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_725);  mul_271 = unsqueeze_725 = None
        unsqueeze_726 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_727 = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_198 = torch.ops.aten.add.Tensor(mul_272, unsqueeze_727);  mul_272 = unsqueeze_727 = None
        add_199 = torch.ops.aten.add.Tensor(add_198, add_192);  add_198 = add_192 = None
        convolution_91 = torch.ops.aten.convolution.default(add_199, arg196_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_199 = arg196_1 = None
        add_200 = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_91 = torch.ops.aten.sqrt.default(add_200);  add_200 = None
        reciprocal_91 = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_273 = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_728 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_729 = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        unsqueeze_730 = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
        unsqueeze_731 = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_729);  convolution_91 = unsqueeze_729 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_733 = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_733);  mul_274 = unsqueeze_733 = None
        unsqueeze_734 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_735 = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_201 = torch.ops.aten.add.Tensor(mul_275, unsqueeze_735);  mul_275 = unsqueeze_735 = None
        clamp_min_61 = torch.ops.aten.clamp_min.default(add_201, 0.0);  add_201 = None
        clamp_max_61 = torch.ops.aten.clamp_max.default(clamp_min_61, 6.0);  clamp_min_61 = None
        convolution_92 = torch.ops.aten.convolution.default(clamp_max_61, arg201_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 576);  clamp_max_61 = arg201_1 = None
        add_202 = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_92 = torch.ops.aten.sqrt.default(add_202);  add_202 = None
        reciprocal_92 = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_276 = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_736 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_737 = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        unsqueeze_738 = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
        unsqueeze_739 = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        sub_92 = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_737);  convolution_92 = unsqueeze_737 = None
        mul_277 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740 = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_741 = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_741);  mul_277 = unsqueeze_741 = None
        unsqueeze_742 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_743 = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_203 = torch.ops.aten.add.Tensor(mul_278, unsqueeze_743);  mul_278 = unsqueeze_743 = None
        clamp_min_62 = torch.ops.aten.clamp_min.default(add_203, 0.0);  add_203 = None
        clamp_max_62 = torch.ops.aten.clamp_max.default(clamp_min_62, 6.0);  clamp_min_62 = None
        convolution_93 = torch.ops.aten.convolution.default(clamp_max_62, arg206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_62 = arg206_1 = None
        add_204 = torch.ops.aten.add.Tensor(arg208_1, 1e-05);  arg208_1 = None
        sqrt_93 = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_93 = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_279 = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_744 = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_745 = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        unsqueeze_746 = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
        unsqueeze_747 = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_745);  convolution_93 = unsqueeze_745 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_749 = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_749);  mul_280 = unsqueeze_749 = None
        unsqueeze_750 = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_751 = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_205 = torch.ops.aten.add.Tensor(mul_281, unsqueeze_751);  mul_281 = unsqueeze_751 = None
        convolution_94 = torch.ops.aten.convolution.default(add_205, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg211_1 = None
        add_206 = torch.ops.aten.add.Tensor(arg213_1, 1e-05);  arg213_1 = None
        sqrt_94 = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_94 = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_282 = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_752 = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_753 = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        unsqueeze_754 = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
        unsqueeze_755 = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        sub_94 = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_753);  convolution_94 = unsqueeze_753 = None
        mul_283 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_757 = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_757);  mul_283 = unsqueeze_757 = None
        unsqueeze_758 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_759 = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_207 = torch.ops.aten.add.Tensor(mul_284, unsqueeze_759);  mul_284 = unsqueeze_759 = None
        clamp_min_63 = torch.ops.aten.clamp_min.default(add_207, 0.0);  add_207 = None
        clamp_max_63 = torch.ops.aten.clamp_max.default(clamp_min_63, 6.0);  clamp_min_63 = None
        convolution_95 = torch.ops.aten.convolution.default(clamp_max_63, arg216_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960);  clamp_max_63 = arg216_1 = None
        add_208 = torch.ops.aten.add.Tensor(arg218_1, 1e-05);  arg218_1 = None
        sqrt_95 = torch.ops.aten.sqrt.default(add_208);  add_208 = None
        reciprocal_95 = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_285 = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_760 = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_761 = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        unsqueeze_762 = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
        unsqueeze_763 = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        sub_95 = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_761);  convolution_95 = unsqueeze_761 = None
        mul_286 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_765 = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_765);  mul_286 = unsqueeze_765 = None
        unsqueeze_766 = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_767 = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_209 = torch.ops.aten.add.Tensor(mul_287, unsqueeze_767);  mul_287 = unsqueeze_767 = None
        clamp_min_64 = torch.ops.aten.clamp_min.default(add_209, 0.0);  add_209 = None
        clamp_max_64 = torch.ops.aten.clamp_max.default(clamp_min_64, 6.0);  clamp_min_64 = None
        convolution_96 = torch.ops.aten.convolution.default(clamp_max_64, arg221_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_64 = arg221_1 = None
        add_210 = torch.ops.aten.add.Tensor(arg223_1, 1e-05);  arg223_1 = None
        sqrt_96 = torch.ops.aten.sqrt.default(add_210);  add_210 = None
        reciprocal_96 = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_288 = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_768 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_769 = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        unsqueeze_770 = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_771 = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        sub_96 = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_769);  convolution_96 = unsqueeze_769 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772 = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_773 = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_773);  mul_289 = unsqueeze_773 = None
        unsqueeze_774 = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_775 = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_211 = torch.ops.aten.add.Tensor(mul_290, unsqueeze_775);  mul_290 = unsqueeze_775 = None
        add_212 = torch.ops.aten.add.Tensor(add_211, add_205);  add_211 = add_205 = None
        convolution_97 = torch.ops.aten.convolution.default(add_212, arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg226_1 = None
        add_213 = torch.ops.aten.add.Tensor(arg228_1, 1e-05);  arg228_1 = None
        sqrt_97 = torch.ops.aten.sqrt.default(add_213);  add_213 = None
        reciprocal_97 = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_291 = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_776 = torch.ops.aten.unsqueeze.default(arg227_1, -1);  arg227_1 = None
        unsqueeze_777 = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        unsqueeze_778 = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
        unsqueeze_779 = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        sub_97 = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_777);  convolution_97 = unsqueeze_777 = None
        mul_292 = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780 = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_781 = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_781);  mul_292 = unsqueeze_781 = None
        unsqueeze_782 = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_783 = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_214 = torch.ops.aten.add.Tensor(mul_293, unsqueeze_783);  mul_293 = unsqueeze_783 = None
        clamp_min_65 = torch.ops.aten.clamp_min.default(add_214, 0.0);  add_214 = None
        clamp_max_65 = torch.ops.aten.clamp_max.default(clamp_min_65, 6.0);  clamp_min_65 = None
        convolution_98 = torch.ops.aten.convolution.default(clamp_max_65, arg231_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960);  clamp_max_65 = arg231_1 = None
        add_215 = torch.ops.aten.add.Tensor(arg233_1, 1e-05);  arg233_1 = None
        sqrt_98 = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_98 = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_294 = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_784 = torch.ops.aten.unsqueeze.default(arg232_1, -1);  arg232_1 = None
        unsqueeze_785 = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        unsqueeze_786 = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
        unsqueeze_787 = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        sub_98 = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_785);  convolution_98 = unsqueeze_785 = None
        mul_295 = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788 = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_789 = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_296 = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_789);  mul_295 = unsqueeze_789 = None
        unsqueeze_790 = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_791 = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_216 = torch.ops.aten.add.Tensor(mul_296, unsqueeze_791);  mul_296 = unsqueeze_791 = None
        clamp_min_66 = torch.ops.aten.clamp_min.default(add_216, 0.0);  add_216 = None
        clamp_max_66 = torch.ops.aten.clamp_max.default(clamp_min_66, 6.0);  clamp_min_66 = None
        convolution_99 = torch.ops.aten.convolution.default(clamp_max_66, arg236_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_66 = arg236_1 = None
        add_217 = torch.ops.aten.add.Tensor(arg238_1, 1e-05);  arg238_1 = None
        sqrt_99 = torch.ops.aten.sqrt.default(add_217);  add_217 = None
        reciprocal_99 = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_297 = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_792 = torch.ops.aten.unsqueeze.default(arg237_1, -1);  arg237_1 = None
        unsqueeze_793 = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        unsqueeze_794 = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
        unsqueeze_795 = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        sub_99 = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_793);  convolution_99 = unsqueeze_793 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796 = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_797 = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_797);  mul_298 = unsqueeze_797 = None
        unsqueeze_798 = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_799 = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_218 = torch.ops.aten.add.Tensor(mul_299, unsqueeze_799);  mul_299 = unsqueeze_799 = None
        add_219 = torch.ops.aten.add.Tensor(add_218, add_212);  add_218 = add_212 = None
        convolution_100 = torch.ops.aten.convolution.default(add_219, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_219 = arg241_1 = None
        add_220 = torch.ops.aten.add.Tensor(arg243_1, 1e-05);  arg243_1 = None
        sqrt_100 = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_100 = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_300 = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_800 = torch.ops.aten.unsqueeze.default(arg242_1, -1);  arg242_1 = None
        unsqueeze_801 = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        unsqueeze_802 = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_803 = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        sub_100 = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_801);  convolution_100 = unsqueeze_801 = None
        mul_301 = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804 = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_805 = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_805);  mul_301 = unsqueeze_805 = None
        unsqueeze_806 = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_807 = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_221 = torch.ops.aten.add.Tensor(mul_302, unsqueeze_807);  mul_302 = unsqueeze_807 = None
        clamp_min_67 = torch.ops.aten.clamp_min.default(add_221, 0.0);  add_221 = None
        clamp_max_67 = torch.ops.aten.clamp_max.default(clamp_min_67, 6.0);  clamp_min_67 = None
        convolution_101 = torch.ops.aten.convolution.default(clamp_max_67, arg246_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 960);  clamp_max_67 = arg246_1 = None
        add_222 = torch.ops.aten.add.Tensor(arg248_1, 1e-05);  arg248_1 = None
        sqrt_101 = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_101 = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_303 = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_808 = torch.ops.aten.unsqueeze.default(arg247_1, -1);  arg247_1 = None
        unsqueeze_809 = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        unsqueeze_810 = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_811 = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        sub_101 = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_809);  convolution_101 = unsqueeze_809 = None
        mul_304 = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812 = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_813 = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_813);  mul_304 = unsqueeze_813 = None
        unsqueeze_814 = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_815 = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_223 = torch.ops.aten.add.Tensor(mul_305, unsqueeze_815);  mul_305 = unsqueeze_815 = None
        clamp_min_68 = torch.ops.aten.clamp_min.default(add_223, 0.0);  add_223 = None
        clamp_max_68 = torch.ops.aten.clamp_max.default(clamp_min_68, 6.0);  clamp_min_68 = None
        convolution_102 = torch.ops.aten.convolution.default(clamp_max_68, arg251_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  clamp_max_68 = arg251_1 = None
        add_224 = torch.ops.aten.add.Tensor(arg253_1, 1e-05);  arg253_1 = None
        sqrt_102 = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_102 = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_306 = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_816 = torch.ops.aten.unsqueeze.default(arg252_1, -1);  arg252_1 = None
        unsqueeze_817 = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        unsqueeze_818 = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
        unsqueeze_819 = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        sub_102 = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_817);  convolution_102 = unsqueeze_817 = None
        mul_307 = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820 = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_821 = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_821);  mul_307 = unsqueeze_821 = None
        unsqueeze_822 = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_823 = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_225 = torch.ops.aten.add.Tensor(mul_308, unsqueeze_823);  mul_308 = unsqueeze_823 = None
        convolution_103 = torch.ops.aten.convolution.default(add_225, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_225 = arg256_1 = None
        add_226 = torch.ops.aten.add.Tensor(arg258_1, 1e-05);  arg258_1 = None
        sqrt_103 = torch.ops.aten.sqrt.default(add_226);  add_226 = None
        reciprocal_103 = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_309 = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_824 = torch.ops.aten.unsqueeze.default(arg257_1, -1);  arg257_1 = None
        unsqueeze_825 = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        unsqueeze_826 = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_827 = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        sub_103 = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_825);  convolution_103 = unsqueeze_825 = None
        mul_310 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828 = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_829 = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_829);  mul_310 = unsqueeze_829 = None
        unsqueeze_830 = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_831 = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_227 = torch.ops.aten.add.Tensor(mul_311, unsqueeze_831);  mul_311 = unsqueeze_831 = None
        clamp_min_69 = torch.ops.aten.clamp_min.default(add_227, 0.0);  add_227 = None
        clamp_max_69 = torch.ops.aten.clamp_max.default(clamp_min_69, 6.0);  clamp_min_69 = None
        mean_1 = torch.ops.aten.mean.dim(clamp_max_69, [-1, -2], True);  clamp_max_69 = None
        view_1 = torch.ops.aten.view.default(mean_1, [8, 1280]);  mean_1 = None
        permute_1 = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg262_1, view_1, permute_1);  arg262_1 = view_1 = permute_1 = None
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
    buf26 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf26, (24, 96, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf27, (24,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf28, (24,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf29, (24,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf30, (24,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf31, (144, 24, 1, 1), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf32, (144,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf33, (144,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf34, (144,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf35, (144,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf36, (144, 1, 3, 3), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf37, (144,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf38, (144,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf39, (144,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf40, (144,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf41, (24, 144, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf42, (24,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf43, (24,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf44, (24,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf45, (24,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf46, (144, 24, 1, 1), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf47, (144,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf48, (144,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf49, (144,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf50, (144,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 5184, device=device(type='cuda', index=0))
    reader.tensor(buf51, (144, 1, 3, 3), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf52, (144,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf53, (144,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf54, (144,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf55, (144,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf56, (32, 144, 1, 1), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf57, (32,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf58, (32,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf59, (32,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf60, (32,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf61, (192, 32, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf62, (192,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf63, (192,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf64, (192,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf65, (192,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf66, (192, 1, 3, 3), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf67, (192,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf68, (192,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf69, (192,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf70, (192,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf71, (32, 192, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf72, (32,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf73, (32,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf74, (32,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf75, (32,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf76, (192, 32, 1, 1), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf77, (192,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf78, (192,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf79, (192,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf80, (192,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf81, (192, 1, 3, 3), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf82, (192,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf83, (192,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf84, (192,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf85, (192,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf86, (32, 192, 1, 1), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf87, (32,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf88, (32,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf89, (32,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf90, (32,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 24576, device=device(type='cuda', index=0))
    reader.tensor(buf91, (192, 32, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf92, (192,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf93, (192,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf94, (192,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf95, (192,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 6912, device=device(type='cuda', index=0))
    reader.tensor(buf96, (192, 1, 3, 3), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf97, (192,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf98, (192,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf99, (192,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf100, (192,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 49152, device=device(type='cuda', index=0))
    reader.tensor(buf101, (64, 192, 1, 1), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf102, (64,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf103, (64,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf104, (64,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf105, (64,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf106, (384, 64, 1, 1), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf107, (384,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf108, (384,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf109, (384,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf110, (384,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf111, (384, 1, 3, 3), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf112, (384,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf113, (384,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf114, (384,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf115, (384,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf116, (64, 384, 1, 1), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf117, (64,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf118, (64,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf119, (64,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf120, (64,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf121, (384, 64, 1, 1), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf122, (384,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf123, (384,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf124, (384,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf125, (384,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf126, (384, 1, 3, 3), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf127, (384,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf128, (384,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf129, (384,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf130, (384,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf131, (64, 384, 1, 1), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf132, (64,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf133, (64,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf134, (64,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf135, (64,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf136, (384, 64, 1, 1), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf137, (384,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf138, (384,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf139, (384,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf140, (384,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf141, (384, 1, 3, 3), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf142, (384,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf143, (384,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf144, (384,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf145, (384,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf146, (64, 384, 1, 1), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf147, (64,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf148, (64,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf149, (64,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf150, (64,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf151, (384, 64, 1, 1), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf152, (384,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf153, (384,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf154, (384,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf155, (384,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf156, (384, 1, 3, 3), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf157, (384,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf158, (384,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf159, (384,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf160, (384,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf161, (96, 384, 1, 1), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf162, (96,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf163, (96,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf164, (96,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf165, (96,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf166, (576, 96, 1, 1), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf167, (576,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf168, (576,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf169, (576,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf170, (576,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf171, (576, 1, 3, 3), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf172, (576,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf173, (576,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf174, (576,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf175, (576,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf176, (96, 576, 1, 1), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf177, (96,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf178, (96,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf179, (96,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf180, (96,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf181, (576, 96, 1, 1), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf182, (576,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf183, (576,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf184, (576,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf185, (576,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf186, (576, 1, 3, 3), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf187, (576,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf188, (576,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf189, (576,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf190, (576,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf191, (96, 576, 1, 1), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf192, (96,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf193, (96,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf194, (96,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf195, (96,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 221184, device=device(type='cuda', index=0))
    reader.tensor(buf196, (576, 96, 1, 1), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf197, (576,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf198, (576,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf199, (576,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf200, (576,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 20736, device=device(type='cuda', index=0))
    reader.tensor(buf201, (576, 1, 3, 3), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf202, (576,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf203, (576,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf204, (576,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf205, (576,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 368640, device=device(type='cuda', index=0))
    reader.tensor(buf206, (160, 576, 1, 1), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf207, (160,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf208, (160,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf209, (160,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf210, (160,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 614400, device=device(type='cuda', index=0))
    reader.tensor(buf211, (960, 160, 1, 1), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf212, (960,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf213, (960,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf214, (960,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf215, (960,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 34560, device=device(type='cuda', index=0))
    reader.tensor(buf216, (960, 1, 3, 3), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf217, (960,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf218, (960,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf219, (960,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf220, (960,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 614400, device=device(type='cuda', index=0))
    reader.tensor(buf221, (160, 960, 1, 1), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf222, (160,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf223, (160,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf224, (160,), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf225, (160,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 614400, device=device(type='cuda', index=0))
    reader.tensor(buf226, (960, 160, 1, 1), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf227, (960,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf228, (960,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf229, (960,), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf230, (960,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 34560, device=device(type='cuda', index=0))
    reader.tensor(buf231, (960, 1, 3, 3), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf232, (960,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf233, (960,), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf234, (960,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf235, (960,), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 614400, device=device(type='cuda', index=0))
    reader.tensor(buf236, (160, 960, 1, 1), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf237, (160,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf238, (160,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf239, (160,), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf240, (160,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 614400, device=device(type='cuda', index=0))
    reader.tensor(buf241, (960, 160, 1, 1), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf242, (960,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf243, (960,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf244, (960,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf245, (960,), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 34560, device=device(type='cuda', index=0))
    reader.tensor(buf246, (960, 1, 3, 3), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf247, (960,), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf248, (960,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf249, (960,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf250, (960,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 1228800, device=device(type='cuda', index=0))
    reader.tensor(buf251, (320, 960, 1, 1), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf252, (320,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf253, (320,), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf254, (320,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 1280, device=device(type='cuda', index=0))
    reader.tensor(buf255, (320,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 1638400, device=device(type='cuda', index=0))
    reader.tensor(buf256, (1280, 320, 1, 1), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf257, (1280,), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf258, (1280,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf259, (1280,), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf260, (1280,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 5120000, device=device(type='cuda', index=0))
    reader.tensor(buf261, (1000, 1280), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf262, (1000,), is_leaf=True)  # arg262_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)