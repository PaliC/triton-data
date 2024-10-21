
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1):
        convolution_41 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_82 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_82);  add_82 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_123 = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_41 = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
        mul_124 = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_83 = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
        relu_41 = torch.ops.aten.relu.default(add_83);  add_83 = None
        convolution_42 = torch.ops.aten.convolution.default(relu_41, arg6_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_41 = arg6_1 = None
        add_84 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_42 = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_126 = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_42 = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
        mul_127 = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_128 = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_85 = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
        relu_42 = torch.ops.aten.relu.default(add_85);  add_85 = None
        convolution_43 = torch.ops.aten.convolution.default(relu_42, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg11_1 = None
        add_86 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_43 = torch.ops.aten.sqrt.default(add_86);  add_86 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_129 = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_43 = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
        mul_130 = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_131 = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_87 = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
        relu_43 = torch.ops.aten.relu.default(add_87);  add_87 = None
        convolution_44 = torch.ops.aten.convolution.default(relu_43, arg16_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_43 = arg16_1 = None
        add_88 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_44 = torch.ops.aten.sqrt.default(add_88);  add_88 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_132 = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_44 = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
        mul_133 = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_89 = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
        relu_44 = torch.ops.aten.relu.default(add_89);  add_89 = None
        convolution_45 = torch.ops.aten.convolution.default(relu_44, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg21_1 = None
        add_90 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_45 = torch.ops.aten.sqrt.default(add_90);  add_90 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_135 = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
        mul_136 = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_91 = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
        relu_45 = torch.ops.aten.relu.default(add_91);  add_91 = None
        convolution_46 = torch.ops.aten.convolution.default(relu_45, arg26_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_45 = arg26_1 = None
        add_92 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_46 = torch.ops.aten.sqrt.default(add_92);  add_92 = None
        reciprocal_46 = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_138 = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_46 = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
        mul_139 = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_93 = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
        relu_46 = torch.ops.aten.relu.default(add_93);  add_93 = None
        cat_6 = torch.ops.aten.cat.default([relu_42, relu_44, relu_46], 1);  relu_42 = relu_44 = relu_46 = None
        convolution_47 = torch.ops.aten.convolution.default(cat_6, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_6 = arg31_1 = None
        add_94 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_47 = torch.ops.aten.sqrt.default(add_94);  add_94 = None
        reciprocal_47 = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_141 = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_47 = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
        mul_142 = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_95 = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
        relu_47 = torch.ops.aten.relu.default(add_95);  add_95 = None
        convolution_48 = torch.ops.aten.convolution.default(relu_47, arg36_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg36_1 = None
        add_96 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_48 = torch.ops.aten.sqrt.default(add_96);  add_96 = None
        reciprocal_48 = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_144 = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_48 = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
        mul_145 = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_97 = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
        relu_48 = torch.ops.aten.relu.default(add_97);  add_97 = None
        convolution_49 = torch.ops.aten.convolution.default(relu_48, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg41_1 = None
        add_98 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_49 = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_49 = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_147 = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_49 = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
        mul_148 = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_99 = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
        relu_49 = torch.ops.aten.relu.default(add_99);  add_99 = None
        convolution_50 = torch.ops.aten.convolution.default(relu_49, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_49 = arg46_1 = None
        add_100 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_50 = torch.ops.aten.sqrt.default(add_100);  add_100 = None
        reciprocal_50 = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_150 = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_50 = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
        mul_151 = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_101 = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
        relu_50 = torch.ops.aten.relu.default(add_101);  add_101 = None
        convolution_51 = torch.ops.aten.convolution.default(relu_50, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg51_1 = None
        add_102 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_51 = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_51 = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_153 = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_51 = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
        mul_154 = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_155 = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_103 = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
        relu_51 = torch.ops.aten.relu.default(add_103);  add_103 = None
        convolution_52 = torch.ops.aten.convolution.default(relu_51, arg56_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_51 = arg56_1 = None
        add_104 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_52 = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_156 = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
        mul_157 = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_105 = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
        relu_52 = torch.ops.aten.relu.default(add_105);  add_105 = None
        cat_7 = torch.ops.aten.cat.default([relu_48, relu_50, relu_52, relu_47], 1);  relu_48 = relu_50 = relu_52 = relu_47 = None
        convolution_53 = torch.ops.aten.convolution.default(cat_7, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_7 = arg61_1 = None
        add_106 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_53 = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_53 = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_159 = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_53 = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_425);  convolution_53 = unsqueeze_425 = None
        mul_160 = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_107 = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
        relu_53 = torch.ops.aten.relu.default(add_107);  add_107 = None
        convolution_54 = torch.ops.aten.convolution.default(relu_53, arg66_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_53 = arg66_1 = None
        add_108 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_54 = torch.ops.aten.sqrt.default(add_108);  add_108 = None
        reciprocal_54 = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_162 = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_54 = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  convolution_54 = unsqueeze_433 = None
        mul_163 = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_109 = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
        relu_54 = torch.ops.aten.relu.default(add_109);  add_109 = None
        convolution_55 = torch.ops.aten.convolution.default(relu_54, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg71_1 = None
        add_110 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_55 = torch.ops.aten.sqrt.default(add_110);  add_110 = None
        reciprocal_55 = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_165 = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_55 = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  convolution_55 = unsqueeze_441 = None
        mul_166 = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_111 = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
        relu_55 = torch.ops.aten.relu.default(add_111);  add_111 = None
        convolution_56 = torch.ops.aten.convolution.default(relu_55, arg76_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_55 = arg76_1 = None
        add_112 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_56 = torch.ops.aten.sqrt.default(add_112);  add_112 = None
        reciprocal_56 = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_168 = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_56 = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_449);  convolution_56 = unsqueeze_449 = None
        mul_169 = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_170 = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_113 = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
        relu_56 = torch.ops.aten.relu.default(add_113);  add_113 = None
        convolution_57 = torch.ops.aten.convolution.default(relu_56, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg81_1 = None
        add_114 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_57 = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_57 = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_171 = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458 = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_459 = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_57 = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_457);  convolution_57 = unsqueeze_457 = None
        mul_172 = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
        unsqueeze_460 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_461 = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_115 = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
        relu_57 = torch.ops.aten.relu.default(add_115);  add_115 = None
        convolution_58 = torch.ops.aten.convolution.default(relu_57, arg86_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_57 = arg86_1 = None
        add_116 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_58 = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_58 = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_174 = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_58 = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_465);  convolution_58 = unsqueeze_465 = None
        mul_175 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_117 = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
        relu_58 = torch.ops.aten.relu.default(add_117);  add_117 = None
        cat_8 = torch.ops.aten.cat.default([relu_54, relu_56, relu_58], 1);  relu_54 = relu_56 = relu_58 = None
        convolution_59 = torch.ops.aten.convolution.default(cat_8, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_8 = arg91_1 = None
        add_118 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_59 = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_59 = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_177 = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_59 = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_473);  convolution_59 = unsqueeze_473 = None
        mul_178 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_119 = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
        relu_59 = torch.ops.aten.relu.default(add_119);  add_119 = None
        convolution_60 = torch.ops.aten.convolution.default(relu_59, arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg96_1 = None
        add_120 = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_60 = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_60 = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_180 = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_60 = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_481);  convolution_60 = unsqueeze_481 = None
        mul_181 = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_182 = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_121 = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
        relu_60 = torch.ops.aten.relu.default(add_121);  add_121 = None
        convolution_61 = torch.ops.aten.convolution.default(relu_60, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg101_1 = None
        add_122 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_61 = torch.ops.aten.sqrt.default(add_122);  add_122 = None
        reciprocal_61 = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_183 = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_61 = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_489);  convolution_61 = unsqueeze_489 = None
        mul_184 = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_123 = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
        relu_61 = torch.ops.aten.relu.default(add_123);  add_123 = None
        convolution_62 = torch.ops.aten.convolution.default(relu_61, arg106_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_61 = arg106_1 = None
        add_124 = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_62 = torch.ops.aten.sqrt.default(add_124);  add_124 = None
        reciprocal_62 = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_186 = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_62 = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_497);  convolution_62 = unsqueeze_497 = None
        mul_187 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_125 = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
        relu_62 = torch.ops.aten.relu.default(add_125);  add_125 = None
        convolution_63 = torch.ops.aten.convolution.default(relu_62, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg111_1 = None
        add_126 = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_63 = torch.ops.aten.sqrt.default(add_126);  add_126 = None
        reciprocal_63 = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_189 = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_63 = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_505);  convolution_63 = unsqueeze_505 = None
        mul_190 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_127 = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
        relu_63 = torch.ops.aten.relu.default(add_127);  add_127 = None
        convolution_64 = torch.ops.aten.convolution.default(relu_63, arg116_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_63 = arg116_1 = None
        add_128 = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_64 = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_64 = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_192 = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_64 = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_513);  convolution_64 = unsqueeze_513 = None
        mul_193 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_129 = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
        relu_64 = torch.ops.aten.relu.default(add_129);  add_129 = None
        cat_9 = torch.ops.aten.cat.default([relu_60, relu_62, relu_64, relu_59], 1);  relu_60 = relu_62 = relu_64 = relu_59 = None
        convolution_65 = torch.ops.aten.convolution.default(cat_9, arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_9 = arg121_1 = None
        add_130 = torch.ops.aten.add.Tensor(arg123_1, 1e-05);  arg123_1 = None
        sqrt_65 = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_65 = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_195 = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_65 = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_521);  convolution_65 = unsqueeze_521 = None
        mul_196 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_131 = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
        relu_65 = torch.ops.aten.relu.default(add_131);  add_131 = None
        convolution_66 = torch.ops.aten.convolution.default(relu_65, arg126_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_65 = arg126_1 = None
        add_132 = torch.ops.aten.add.Tensor(arg128_1, 1e-05);  arg128_1 = None
        sqrt_66 = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_66 = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_198 = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_66 = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_529);  convolution_66 = unsqueeze_529 = None
        mul_199 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_133 = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
        relu_66 = torch.ops.aten.relu.default(add_133);  add_133 = None
        convolution_67 = torch.ops.aten.convolution.default(relu_66, arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg131_1 = None
        add_134 = torch.ops.aten.add.Tensor(arg133_1, 1e-05);  arg133_1 = None
        sqrt_67 = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_67 = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_201 = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_537);  convolution_67 = unsqueeze_537 = None
        mul_202 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_543 = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_135 = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
        relu_67 = torch.ops.aten.relu.default(add_135);  add_135 = None
        convolution_68 = torch.ops.aten.convolution.default(relu_67, arg136_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_67 = arg136_1 = None
        add_136 = torch.ops.aten.add.Tensor(arg138_1, 1e-05);  arg138_1 = None
        sqrt_68 = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_68 = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_204 = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_545 = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_68 = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_545);  convolution_68 = unsqueeze_545 = None
        mul_205 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_206 = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_137 = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
        relu_68 = torch.ops.aten.relu.default(add_137);  add_137 = None
        convolution_69 = torch.ops.aten.convolution.default(relu_68, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg141_1 = None
        add_138 = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_69 = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_69 = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_207 = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_69 = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_553);  convolution_69 = unsqueeze_553 = None
        mul_208 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_139 = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
        relu_69 = torch.ops.aten.relu.default(add_139);  add_139 = None
        convolution_70 = torch.ops.aten.convolution.default(relu_69, arg146_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_69 = arg146_1 = None
        add_140 = torch.ops.aten.add.Tensor(arg148_1, 1e-05);  arg148_1 = None
        sqrt_70 = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_70 = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_210 = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_70 = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_561);  convolution_70 = unsqueeze_561 = None
        mul_211 = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_141 = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
        relu_70 = torch.ops.aten.relu.default(add_141);  add_141 = None
        cat_10 = torch.ops.aten.cat.default([relu_66, relu_68, relu_70], 1);  relu_66 = relu_68 = relu_70 = None
        convolution_71 = torch.ops.aten.convolution.default(cat_10, arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_10 = arg151_1 = None
        add_142 = torch.ops.aten.add.Tensor(arg153_1, 1e-05);  arg153_1 = None
        sqrt_71 = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_71 = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_213 = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_71 = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_569);  convolution_71 = unsqueeze_569 = None
        mul_214 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_215 = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_143 = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
        relu_71 = torch.ops.aten.relu.default(add_143);  add_143 = None
        convolution_72 = torch.ops.aten.convolution.default(relu_71, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg156_1 = None
        add_144 = torch.ops.aten.add.Tensor(arg158_1, 1e-05);  arg158_1 = None
        sqrt_72 = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_72 = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_216 = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_72 = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_577);  convolution_72 = unsqueeze_577 = None
        mul_217 = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_218 = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_145 = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
        relu_72 = torch.ops.aten.relu.default(add_145);  add_145 = None
        convolution_73 = torch.ops.aten.convolution.default(relu_72, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg161_1 = None
        add_146 = torch.ops.aten.add.Tensor(arg163_1, 1e-05);  arg163_1 = None
        sqrt_73 = torch.ops.aten.sqrt.default(add_146);  add_146 = None
        reciprocal_73 = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_219 = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_73 = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_585);  convolution_73 = unsqueeze_585 = None
        mul_220 = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_147 = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
        relu_73 = torch.ops.aten.relu.default(add_147);  add_147 = None
        convolution_74 = torch.ops.aten.convolution.default(relu_73, arg166_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_73 = arg166_1 = None
        add_148 = torch.ops.aten.add.Tensor(arg168_1, 1e-05);  arg168_1 = None
        sqrt_74 = torch.ops.aten.sqrt.default(add_148);  add_148 = None
        reciprocal_74 = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_222 = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(arg167_1, -1);  arg167_1 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_74 = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_593);  convolution_74 = unsqueeze_593 = None
        mul_223 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_224 = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_149 = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
        relu_74 = torch.ops.aten.relu.default(add_149);  add_149 = None
        convolution_75 = torch.ops.aten.convolution.default(relu_74, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg171_1 = None
        add_150 = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_75 = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_75 = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_225 = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_75 = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_601);  convolution_75 = unsqueeze_601 = None
        mul_226 = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604 = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_605 = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_227 = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_151 = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
        relu_75 = torch.ops.aten.relu.default(add_151);  add_151 = None
        convolution_76 = torch.ops.aten.convolution.default(relu_75, arg176_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_75 = arg176_1 = None
        add_152 = torch.ops.aten.add.Tensor(arg178_1, 1e-05);  arg178_1 = None
        sqrt_76 = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_76 = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_228 = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_608 = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_609 = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        unsqueeze_610 = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_611 = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        sub_76 = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_609);  convolution_76 = unsqueeze_609 = None
        mul_229 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612 = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_613 = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
        unsqueeze_614 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_615 = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_153 = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
        relu_76 = torch.ops.aten.relu.default(add_153);  add_153 = None
        cat_11 = torch.ops.aten.cat.default([relu_72, relu_74, relu_76, relu_71], 1);  relu_72 = relu_74 = relu_76 = relu_71 = None
        convolution_77 = torch.ops.aten.convolution.default(cat_11, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  cat_11 = arg181_1 = None
        add_154 = torch.ops.aten.add.Tensor(arg183_1, 1e-05);  arg183_1 = None
        sqrt_77 = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_77 = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_231 = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_616 = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_617 = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        unsqueeze_618 = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
        unsqueeze_619 = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        sub_77 = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_617);  convolution_77 = unsqueeze_617 = None
        mul_232 = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620 = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_621 = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
        unsqueeze_622 = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_623 = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_155 = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
        relu_77 = torch.ops.aten.relu.default(add_155);  add_155 = None
        convolution_78 = torch.ops.aten.convolution.default(relu_77, arg186_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_77 = arg186_1 = None
        add_156 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_78 = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_78 = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_234 = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_624 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_625 = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        unsqueeze_626 = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
        unsqueeze_627 = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        sub_78 = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_625);  convolution_78 = unsqueeze_625 = None
        mul_235 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_629 = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
        unsqueeze_630 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_631 = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_157 = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
        relu_78 = torch.ops.aten.relu.default(add_157);  add_157 = None
        convolution_79 = torch.ops.aten.convolution.default(relu_78, arg191_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_78 = arg191_1 = None
        add_158 = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_79 = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_79 = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_237 = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_632 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_633 = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        unsqueeze_634 = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_635 = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        sub_79 = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_633);  convolution_79 = unsqueeze_633 = None
        mul_238 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_637 = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
        unsqueeze_638 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_639 = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_159 = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
        relu_79 = torch.ops.aten.relu.default(add_159);  add_159 = None
        convolution_80 = torch.ops.aten.convolution.default(relu_79, arg196_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_79 = arg196_1 = None
        add_160 = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_80 = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_80 = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_240 = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_640 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_641 = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        unsqueeze_642 = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_643 = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        sub_80 = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_641);  convolution_80 = unsqueeze_641 = None
        mul_241 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_645 = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
        unsqueeze_646 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_647 = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_161 = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
        relu_80 = torch.ops.aten.relu.default(add_161);  add_161 = None
        convolution_81 = torch.ops.aten.convolution.default(relu_80, arg201_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_80 = arg201_1 = None
        add_162 = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_81 = torch.ops.aten.sqrt.default(add_162);  add_162 = None
        reciprocal_81 = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_243 = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_648 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_649 = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        unsqueeze_650 = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_651 = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        sub_81 = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_649);  convolution_81 = unsqueeze_649 = None
        mul_244 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652 = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_653 = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
        unsqueeze_654 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_655 = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_163 = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
        relu_81 = torch.ops.aten.relu.default(add_163);  add_163 = None
        mean_1 = torch.ops.aten.mean.dim(relu_81, [-1, -2], True);  relu_81 = None
        view_1 = torch.ops.aten.view.default(mean_1, [8, 1024]);  mean_1 = None
        permute_1 = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg207_1, view_1, permute_1);  arg207_1 = view_1 = permute_1 = None
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
    buf6 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 32, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64, 64, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf16, (32, 64, 3, 3), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf17, (32,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf18, (32,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf19, (32,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf20, (32,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf21, (64, 32, 1, 1), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf22, (64,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf23, (64,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf25, (64,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf26, (32, 64, 3, 3), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf27, (32,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf28, (32,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf29, (32,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf30, (32,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf31, (64, 128, 1, 1), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf32, (64,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf33, (64,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf34, (64,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf35, (64,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf36, (64, 64, 3, 3), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf37, (64,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf38, (64,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf39, (64,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf40, (64,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf41, (64, 64, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf42, (64,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf43, (64,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf46, (32, 64, 3, 3), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf47, (32,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf48, (32,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf49, (32,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf50, (32,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf51, (64, 32, 1, 1), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf52, (64,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf53, (64,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf54, (64,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf55, (64,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf56, (32, 64, 3, 3), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf57, (32,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf58, (32,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf59, (32,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf60, (32,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128, 192, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf65, (128,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf66, (144, 128, 3, 3), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf67, (144,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf68, (144,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf69, (144,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf70, (144,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf71, (144, 144, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf72, (144,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf73, (144,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf74, (144,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf75, (144,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf76, (72, 144, 3, 3), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf77, (72,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf78, (72,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf79, (72,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf80, (72,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 41472, device=device(type='cuda', index=0))
    reader.tensor(buf81, (144, 72, 1, 1), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf82, (144,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf83, (144,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf84, (144,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf85, (144,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf86, (72, 144, 3, 3), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf87, (72,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf88, (72,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf89, (72,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf90, (72,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf91, (144, 288, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf92, (144,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf93, (144,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf94, (144,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf95, (144,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 746496, device=device(type='cuda', index=0))
    reader.tensor(buf96, (144, 144, 3, 3), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf97, (144,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf98, (144,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf99, (144,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf100, (144,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf101, (144, 144, 1, 1), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf102, (144,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf103, (144,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf104, (144,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf105, (144,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf106, (72, 144, 3, 3), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf107, (72,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf108, (72,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf109, (72,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf110, (72,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 41472, device=device(type='cuda', index=0))
    reader.tensor(buf111, (144, 72, 1, 1), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf112, (144,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf113, (144,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf114, (144,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf115, (144,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 373248, device=device(type='cuda', index=0))
    reader.tensor(buf116, (72, 144, 3, 3), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf117, (72,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf118, (72,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf119, (72,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf120, (72,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 497664, device=device(type='cuda', index=0))
    reader.tensor(buf121, (288, 432, 1, 1), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf122, (288,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf123, (288,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf124, (288,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf125, (288,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 3151872, device=device(type='cuda', index=0))
    reader.tensor(buf126, (304, 288, 3, 3), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf127, (304,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf128, (304,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf129, (304,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf130, (304,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 369664, device=device(type='cuda', index=0))
    reader.tensor(buf131, (304, 304, 1, 1), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf132, (304,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf133, (304,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf134, (304,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf135, (304,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1663488, device=device(type='cuda', index=0))
    reader.tensor(buf136, (152, 304, 3, 3), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf137, (152,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf138, (152,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf139, (152,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf140, (152,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 184832, device=device(type='cuda', index=0))
    reader.tensor(buf141, (304, 152, 1, 1), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf142, (304,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf143, (304,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf144, (304,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf145, (304,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1663488, device=device(type='cuda', index=0))
    reader.tensor(buf146, (152, 304, 3, 3), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf147, (152,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf148, (152,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf149, (152,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf150, (152,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 739328, device=device(type='cuda', index=0))
    reader.tensor(buf151, (304, 608, 1, 1), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf152, (304,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf153, (304,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf154, (304,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf155, (304,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 3326976, device=device(type='cuda', index=0))
    reader.tensor(buf156, (304, 304, 3, 3), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf157, (304,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf158, (304,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf159, (304,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf160, (304,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 369664, device=device(type='cuda', index=0))
    reader.tensor(buf161, (304, 304, 1, 1), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf162, (304,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf163, (304,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf164, (304,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf165, (304,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1663488, device=device(type='cuda', index=0))
    reader.tensor(buf166, (152, 304, 3, 3), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf167, (152,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf168, (152,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf169, (152,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf170, (152,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 184832, device=device(type='cuda', index=0))
    reader.tensor(buf171, (304, 152, 1, 1), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf172, (304,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf173, (304,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf174, (304,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 1216, device=device(type='cuda', index=0))
    reader.tensor(buf175, (304,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 1663488, device=device(type='cuda', index=0))
    reader.tensor(buf176, (152, 304, 3, 3), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf177, (152,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf178, (152,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf179, (152,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 608, device=device(type='cuda', index=0))
    reader.tensor(buf180, (152,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 1751040, device=device(type='cuda', index=0))
    reader.tensor(buf181, (480, 912, 1, 1), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf182, (480,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf183, (480,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf184, (480,), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf185, (480,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 16588800, device=device(type='cuda', index=0))
    reader.tensor(buf186, (960, 480, 3, 3), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf187, (960,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf188, (960,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf189, (960,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 3840, device=device(type='cuda', index=0))
    reader.tensor(buf190, (960,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 35389440, device=device(type='cuda', index=0))
    reader.tensor(buf191, (1024, 960, 3, 3), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf192, (1024,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf193, (1024,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf194, (1024,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf195, (1024,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 47185920, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1280, 1024, 3, 3), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf197, (1280,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf198, (1280,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf199, (1280,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf200, (1280,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 5242880, device=device(type='cuda', index=0))
    reader.tensor(buf201, (1024, 1280, 1, 1), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf202, (1024,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf203, (1024,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf204, (1024,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf205, (1024,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 4096000, device=device(type='cuda', index=0))
    reader.tensor(buf206, (1000, 1024), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf207, (1000,), is_leaf=True)  # arg207_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)