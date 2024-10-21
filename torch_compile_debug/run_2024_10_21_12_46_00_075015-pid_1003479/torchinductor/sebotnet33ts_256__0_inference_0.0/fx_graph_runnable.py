
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1):
        convolution_50 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_94 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_38 = torch.ops.aten.sqrt.default(add_94);  add_94 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_158 = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(mul_158, -1);  mul_158 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_42 = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_305);  convolution_50 = unsqueeze_305 = None
        mul_159 = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_307);  sub_42 = unsqueeze_307 = None
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_309);  mul_159 = unsqueeze_309 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_95 = torch.ops.aten.add.Tensor(mul_160, unsqueeze_311);  mul_160 = unsqueeze_311 = None
        sigmoid_40 = torch.ops.aten.sigmoid.default(add_95)
        mul_161 = torch.ops.aten.mul.Tensor(add_95, sigmoid_40);  add_95 = sigmoid_40 = None
        convolution_51 = torch.ops.aten.convolution.default(mul_161, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_161 = arg6_1 = None
        add_96 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_39 = torch.ops.aten.sqrt.default(add_96);  add_96 = None
        reciprocal_39 = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_162 = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_43 = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_313);  convolution_51 = unsqueeze_313 = None
        mul_163 = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_315);  sub_43 = unsqueeze_315 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_317);  mul_163 = unsqueeze_317 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_97 = torch.ops.aten.add.Tensor(mul_164, unsqueeze_319);  mul_164 = unsqueeze_319 = None
        sigmoid_41 = torch.ops.aten.sigmoid.default(add_97)
        mul_165 = torch.ops.aten.mul.Tensor(add_97, sigmoid_41);  add_97 = sigmoid_41 = None
        convolution_52 = torch.ops.aten.convolution.default(mul_165, arg11_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_165 = arg11_1 = None
        add_98 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_40 = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_40 = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_166 = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(mul_166, -1);  mul_166 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_44 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_321);  convolution_52 = unsqueeze_321 = None
        mul_167 = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_323);  sub_44 = unsqueeze_323 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_168 = torch.ops.aten.mul.Tensor(mul_167, unsqueeze_325);  mul_167 = unsqueeze_325 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_99 = torch.ops.aten.add.Tensor(mul_168, unsqueeze_327);  mul_168 = unsqueeze_327 = None
        sigmoid_42 = torch.ops.aten.sigmoid.default(add_99)
        mul_169 = torch.ops.aten.mul.Tensor(add_99, sigmoid_42);  add_99 = sigmoid_42 = None
        convolution_53 = torch.ops.aten.convolution.default(mul_169, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg16_1 = None
        add_100 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_100);  add_100 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_170 = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_170, -1);  mul_170 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_329);  convolution_53 = unsqueeze_329 = None
        mul_171 = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_331);  sub_45 = unsqueeze_331 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_171, unsqueeze_333);  mul_171 = unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_101 = torch.ops.aten.add.Tensor(mul_172, unsqueeze_335);  mul_172 = unsqueeze_335 = None
        sigmoid_43 = torch.ops.aten.sigmoid.default(add_101)
        mul_173 = torch.ops.aten.mul.Tensor(add_101, sigmoid_43);  add_101 = sigmoid_43 = None
        convolution_54 = torch.ops.aten.convolution.default(mul_173, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_173 = arg21_1 = None
        add_102 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_42 = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_174 = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_46 = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_337);  convolution_54 = unsqueeze_337 = None
        mul_175 = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_339);  sub_46 = unsqueeze_339 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_341);  mul_175 = unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_103 = torch.ops.aten.add.Tensor(mul_176, unsqueeze_343);  mul_176 = unsqueeze_343 = None
        sigmoid_44 = torch.ops.aten.sigmoid.default(add_103)
        mul_177 = torch.ops.aten.mul.Tensor(add_103, sigmoid_44);  add_103 = sigmoid_44 = None
        mean_7 = torch.ops.aten.mean.dim(mul_177, [2, 3], True)
        convolution_55 = torch.ops.aten.convolution.default(mean_7, arg26_1, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_7 = arg26_1 = arg27_1 = None
        relu_6 = torch.ops.aten.relu.default(convolution_55);  convolution_55 = None
        convolution_56 = torch.ops.aten.convolution.default(relu_6, arg28_1, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_6 = arg28_1 = arg29_1 = None
        sigmoid_45 = torch.ops.aten.sigmoid.default(convolution_56);  convolution_56 = None
        mul_178 = torch.ops.aten.mul.Tensor(mul_177, sigmoid_45);  mul_177 = sigmoid_45 = None
        convolution_57 = torch.ops.aten.convolution.default(mul_178, arg30_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_178 = arg30_1 = None
        add_104 = torch.ops.aten.add.Tensor(arg32_1, 1e-05);  arg32_1 = None
        sqrt_43 = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_179 = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_47 = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_345);  convolution_57 = unsqueeze_345 = None
        mul_180 = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_347);  sub_47 = unsqueeze_347 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_349);  mul_180 = unsqueeze_349 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_105 = torch.ops.aten.add.Tensor(mul_181, unsqueeze_351);  mul_181 = unsqueeze_351 = None
        convolution_58 = torch.ops.aten.convolution.default(mul_169, arg35_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_169 = arg35_1 = None
        add_106 = torch.ops.aten.add.Tensor(arg37_1, 1e-05);  arg37_1 = None
        sqrt_44 = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_182 = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_48 = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_353);  convolution_58 = unsqueeze_353 = None
        mul_183 = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_355);  sub_48 = unsqueeze_355 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_357);  mul_183 = unsqueeze_357 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_107 = torch.ops.aten.add.Tensor(mul_184, unsqueeze_359);  mul_184 = unsqueeze_359 = None
        add_108 = torch.ops.aten.add.Tensor(add_105, add_107);  add_105 = add_107 = None
        sigmoid_46 = torch.ops.aten.sigmoid.default(add_108)
        mul_185 = torch.ops.aten.mul.Tensor(add_108, sigmoid_46);  add_108 = sigmoid_46 = None
        convolution_59 = torch.ops.aten.convolution.default(mul_185, arg40_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg40_1 = None
        add_109 = torch.ops.aten.add.Tensor(arg42_1, 1e-05);  arg42_1 = None
        sqrt_45 = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_186 = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_49 = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_361);  convolution_59 = unsqueeze_361 = None
        mul_187 = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_363);  sub_49 = unsqueeze_363 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_365);  mul_187 = unsqueeze_365 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_110 = torch.ops.aten.add.Tensor(mul_188, unsqueeze_367);  mul_188 = unsqueeze_367 = None
        sigmoid_47 = torch.ops.aten.sigmoid.default(add_110)
        mul_189 = torch.ops.aten.mul.Tensor(add_110, sigmoid_47);  add_110 = sigmoid_47 = None
        convolution_60 = torch.ops.aten.convolution.default(mul_189, arg45_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_189 = arg45_1 = None
        add_111 = torch.ops.aten.add.Tensor(arg47_1, 1e-05);  arg47_1 = None
        sqrt_46 = torch.ops.aten.sqrt.default(add_111);  add_111 = None
        reciprocal_46 = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_190 = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(mul_190, -1);  mul_190 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_50 = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_369);  convolution_60 = unsqueeze_369 = None
        mul_191 = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_371);  sub_50 = unsqueeze_371 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_192 = torch.ops.aten.mul.Tensor(mul_191, unsqueeze_373);  mul_191 = unsqueeze_373 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_112 = torch.ops.aten.add.Tensor(mul_192, unsqueeze_375);  mul_192 = unsqueeze_375 = None
        sigmoid_48 = torch.ops.aten.sigmoid.default(add_112)
        mul_193 = torch.ops.aten.mul.Tensor(add_112, sigmoid_48);  add_112 = sigmoid_48 = None
        mean_8 = torch.ops.aten.mean.dim(mul_193, [2, 3], True)
        convolution_61 = torch.ops.aten.convolution.default(mean_8, arg50_1, arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_8 = arg50_1 = arg51_1 = None
        relu_7 = torch.ops.aten.relu.default(convolution_61);  convolution_61 = None
        convolution_62 = torch.ops.aten.convolution.default(relu_7, arg52_1, arg53_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_7 = arg52_1 = arg53_1 = None
        sigmoid_49 = torch.ops.aten.sigmoid.default(convolution_62);  convolution_62 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_193, sigmoid_49);  mul_193 = sigmoid_49 = None
        convolution_63 = torch.ops.aten.convolution.default(mul_194, arg54_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_194 = arg54_1 = None
        add_113 = torch.ops.aten.add.Tensor(arg56_1, 1e-05);  arg56_1 = None
        sqrt_47 = torch.ops.aten.sqrt.default(add_113);  add_113 = None
        reciprocal_47 = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_195 = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_51 = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_377);  convolution_63 = unsqueeze_377 = None
        mul_196 = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_379);  sub_51 = unsqueeze_379 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_381);  mul_196 = unsqueeze_381 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_114 = torch.ops.aten.add.Tensor(mul_197, unsqueeze_383);  mul_197 = unsqueeze_383 = None
        add_115 = torch.ops.aten.add.Tensor(add_114, mul_185);  add_114 = mul_185 = None
        sigmoid_50 = torch.ops.aten.sigmoid.default(add_115)
        mul_198 = torch.ops.aten.mul.Tensor(add_115, sigmoid_50);  add_115 = sigmoid_50 = None
        convolution_64 = torch.ops.aten.convolution.default(mul_198, arg59_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg59_1 = None
        add_116 = torch.ops.aten.add.Tensor(arg61_1, 1e-05);  arg61_1 = None
        sqrt_48 = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_48 = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_199 = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(mul_199, -1);  mul_199 = None
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_385);  convolution_64 = unsqueeze_385 = None
        mul_200 = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_387);  sub_52 = unsqueeze_387 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_389);  mul_200 = unsqueeze_389 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(arg63_1, -1);  arg63_1 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_117 = torch.ops.aten.add.Tensor(mul_201, unsqueeze_391);  mul_201 = unsqueeze_391 = None
        sigmoid_51 = torch.ops.aten.sigmoid.default(add_117)
        mul_202 = torch.ops.aten.mul.Tensor(add_117, sigmoid_51);  add_117 = sigmoid_51 = None
        convolution_65 = torch.ops.aten.convolution.default(mul_202, arg64_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_202 = arg64_1 = None
        add_118 = torch.ops.aten.add.Tensor(arg66_1, 1e-05);  arg66_1 = None
        sqrt_49 = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_49 = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_203 = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(mul_203, -1);  mul_203 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_53 = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_393);  convolution_65 = unsqueeze_393 = None
        mul_204 = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_395);  sub_53 = unsqueeze_395 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_397);  mul_204 = unsqueeze_397 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_119 = torch.ops.aten.add.Tensor(mul_205, unsqueeze_399);  mul_205 = unsqueeze_399 = None
        sigmoid_52 = torch.ops.aten.sigmoid.default(add_119)
        mul_206 = torch.ops.aten.mul.Tensor(add_119, sigmoid_52);  add_119 = sigmoid_52 = None
        mean_9 = torch.ops.aten.mean.dim(mul_206, [2, 3], True)
        convolution_66 = torch.ops.aten.convolution.default(mean_9, arg69_1, arg70_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_9 = arg69_1 = arg70_1 = None
        relu_8 = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None
        convolution_67 = torch.ops.aten.convolution.default(relu_8, arg71_1, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg71_1 = arg72_1 = None
        sigmoid_53 = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
        mul_207 = torch.ops.aten.mul.Tensor(mul_206, sigmoid_53);  mul_206 = sigmoid_53 = None
        convolution_68 = torch.ops.aten.convolution.default(mul_207, arg73_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_207 = arg73_1 = None
        add_120 = torch.ops.aten.add.Tensor(arg75_1, 1e-05);  arg75_1 = None
        sqrt_50 = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_50 = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_208 = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_54 = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_401);  convolution_68 = unsqueeze_401 = None
        mul_209 = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_403);  sub_54 = unsqueeze_403 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(arg76_1, -1);  arg76_1 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_210 = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_405);  mul_209 = unsqueeze_405 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_121 = torch.ops.aten.add.Tensor(mul_210, unsqueeze_407);  mul_210 = unsqueeze_407 = None
        convolution_69 = torch.ops.aten.convolution.default(mul_198, arg78_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_198 = arg78_1 = None
        add_122 = torch.ops.aten.add.Tensor(arg80_1, 1e-05);  arg80_1 = None
        sqrt_51 = torch.ops.aten.sqrt.default(add_122);  add_122 = None
        reciprocal_51 = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_211 = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(mul_211, -1);  mul_211 = None
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_55 = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_409);  convolution_69 = unsqueeze_409 = None
        mul_212 = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_411);  sub_55 = unsqueeze_411 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_213 = torch.ops.aten.mul.Tensor(mul_212, unsqueeze_413);  mul_212 = unsqueeze_413 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_123 = torch.ops.aten.add.Tensor(mul_213, unsqueeze_415);  mul_213 = unsqueeze_415 = None
        add_124 = torch.ops.aten.add.Tensor(add_121, add_123);  add_121 = add_123 = None
        sigmoid_54 = torch.ops.aten.sigmoid.default(add_124)
        mul_214 = torch.ops.aten.mul.Tensor(add_124, sigmoid_54);  add_124 = sigmoid_54 = None
        convolution_70 = torch.ops.aten.convolution.default(mul_214, arg83_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg83_1 = None
        add_125 = torch.ops.aten.add.Tensor(arg85_1, 1e-05);  arg85_1 = None
        sqrt_52 = torch.ops.aten.sqrt.default(add_125);  add_125 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_215 = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(mul_215, -1);  mul_215 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_56 = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_417);  convolution_70 = unsqueeze_417 = None
        mul_216 = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_419);  sub_56 = unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_216, unsqueeze_421);  mul_216 = unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_126 = torch.ops.aten.add.Tensor(mul_217, unsqueeze_423);  mul_217 = unsqueeze_423 = None
        sigmoid_55 = torch.ops.aten.sigmoid.default(add_126)
        mul_218 = torch.ops.aten.mul.Tensor(add_126, sigmoid_55);  add_126 = sigmoid_55 = None
        convolution_71 = torch.ops.aten.convolution.default(mul_218, arg88_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_218 = arg88_1 = None
        add_127 = torch.ops.aten.add.Tensor(arg90_1, 1e-05);  arg90_1 = None
        sqrt_53 = torch.ops.aten.sqrt.default(add_127);  add_127 = None
        reciprocal_53 = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_219 = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_57 = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_425);  convolution_71 = unsqueeze_425 = None
        mul_220 = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_427);  sub_57 = unsqueeze_427 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_429);  mul_220 = unsqueeze_429 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_128 = torch.ops.aten.add.Tensor(mul_221, unsqueeze_431);  mul_221 = unsqueeze_431 = None
        sigmoid_56 = torch.ops.aten.sigmoid.default(add_128)
        mul_222 = torch.ops.aten.mul.Tensor(add_128, sigmoid_56);  add_128 = sigmoid_56 = None
        mean_10 = torch.ops.aten.mean.dim(mul_222, [2, 3], True)
        convolution_72 = torch.ops.aten.convolution.default(mean_10, arg93_1, arg94_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_10 = arg93_1 = arg94_1 = None
        relu_9 = torch.ops.aten.relu.default(convolution_72);  convolution_72 = None
        convolution_73 = torch.ops.aten.convolution.default(relu_9, arg95_1, arg96_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg95_1 = arg96_1 = None
        sigmoid_57 = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_222, sigmoid_57);  mul_222 = sigmoid_57 = None
        convolution_74 = torch.ops.aten.convolution.default(mul_223, arg97_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_223 = arg97_1 = None
        add_129 = torch.ops.aten.add.Tensor(arg99_1, 1e-05);  arg99_1 = None
        sqrt_54 = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_54 = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_224 = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(mul_224, -1);  mul_224 = None
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_58 = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_433);  convolution_74 = unsqueeze_433 = None
        mul_225 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_435);  sub_58 = unsqueeze_435 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_226 = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_437);  mul_225 = unsqueeze_437 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_130 = torch.ops.aten.add.Tensor(mul_226, unsqueeze_439);  mul_226 = unsqueeze_439 = None
        add_131 = torch.ops.aten.add.Tensor(add_130, mul_214);  add_130 = mul_214 = None
        sigmoid_58 = torch.ops.aten.sigmoid.default(add_131)
        mul_227 = torch.ops.aten.mul.Tensor(add_131, sigmoid_58);  add_131 = sigmoid_58 = None
        convolution_75 = torch.ops.aten.convolution.default(mul_227, arg102_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg102_1 = None
        add_132 = torch.ops.aten.add.Tensor(arg104_1, 1e-05);  arg104_1 = None
        sqrt_55 = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_55 = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_228 = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(arg103_1, -1);  arg103_1 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_59 = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_441);  convolution_75 = unsqueeze_441 = None
        mul_229 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_443);  sub_59 = unsqueeze_443 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_445);  mul_229 = unsqueeze_445 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(arg106_1, -1);  arg106_1 = None
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_133 = torch.ops.aten.add.Tensor(mul_230, unsqueeze_447);  mul_230 = unsqueeze_447 = None
        sigmoid_59 = torch.ops.aten.sigmoid.default(add_133)
        mul_231 = torch.ops.aten.mul.Tensor(add_133, sigmoid_59);  add_133 = sigmoid_59 = None
        convolution_76 = torch.ops.aten.convolution.default(mul_231, arg107_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_231 = arg107_1 = None
        split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(convolution_76, [128, 128, 128], 1);  convolution_76 = None
        getitem_12 = split_with_sizes_4[0]
        getitem_13 = split_with_sizes_4[1]
        getitem_14 = split_with_sizes_4[2];  split_with_sizes_4 = None
        clone_29 = torch.ops.aten.clone.default(getitem_12, memory_format = torch.contiguous_format);  getitem_12 = None
        view_97 = torch.ops.aten.view.default(clone_29, [32, 32, 1024]);  clone_29 = None
        permute_33 = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        clone_30 = torch.ops.aten.clone.default(getitem_13, memory_format = torch.contiguous_format);  getitem_13 = None
        view_98 = torch.ops.aten.view.default(clone_30, [32, 32, 1024]);  clone_30 = None
        clone_31 = torch.ops.aten.clone.default(getitem_14, memory_format = torch.contiguous_format);  getitem_14 = None
        view_99 = torch.ops.aten.view.default(clone_31, [32, 32, 1024]);  clone_31 = None
        permute_34 = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        expand_24 = torch.ops.aten.expand.default(permute_33, [32, 1024, 32])
        expand_25 = torch.ops.aten.expand.default(view_98, [32, 32, 1024]);  view_98 = None
        bmm_8 = torch.ops.aten.bmm.default(expand_24, expand_25);  expand_24 = expand_25 = None
        mul_232 = torch.ops.aten.mul.Tensor(bmm_8, 0.1767766952966369);  bmm_8 = None
        view_103 = torch.ops.aten.view.default(permute_33, [32, 32, 32, 32]);  permute_33 = None
        permute_35 = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
        clone_32 = torch.ops.aten.clone.default(view_103, memory_format = torch.contiguous_format)
        view_104 = torch.ops.aten.view.default(clone_32, [32768, 32]);  clone_32 = None
        mm_8 = torch.ops.aten.mm.default(view_104, permute_35);  view_104 = permute_35 = None
        view_105 = torch.ops.aten.view.default(mm_8, [32, 32, 32, 63]);  mm_8 = None
        view_106 = torch.ops.aten.view.default(view_105, [-1, 32, 63]);  view_105 = None
        constant_pad_nd_16 = torch.ops.aten.constant_pad_nd.default(view_106, [0, 1], 0.0);  view_106 = None
        view_107 = torch.ops.aten.view.default(constant_pad_nd_16, [1024, 2048]);  constant_pad_nd_16 = None
        constant_pad_nd_17 = torch.ops.aten.constant_pad_nd.default(view_107, [0, 31], 0.0);  view_107 = None
        view_108 = torch.ops.aten.view.default(constant_pad_nd_17, [-1, 33, 63]);  constant_pad_nd_17 = None
        slice_26 = torch.ops.aten.slice.Tensor(view_108, 1, 0, 32);  view_108 = None
        slice_27 = torch.ops.aten.slice.Tensor(slice_26, 2, 31, 9223372036854775807);  slice_26 = None
        view_109 = torch.ops.aten.view.default(slice_27, [32, 32, 1, 32, 32]);  slice_27 = None
        expand_26 = torch.ops.aten.expand.default(view_109, [-1, -1, 32, -1, -1]);  view_109 = None
        permute_36 = torch.ops.aten.permute.default(expand_26, [0, 1, 3, 2, 4]);  expand_26 = None
        permute_37 = torch.ops.aten.permute.default(view_103, [0, 2, 1, 3]);  view_103 = None
        permute_38 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        clone_33 = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_110 = torch.ops.aten.view.default(clone_33, [32768, 32]);  clone_33 = None
        mm_9 = torch.ops.aten.mm.default(view_110, permute_38);  view_110 = permute_38 = None
        view_111 = torch.ops.aten.view.default(mm_9, [32, 32, 32, 63]);  mm_9 = None
        view_112 = torch.ops.aten.view.default(view_111, [-1, 32, 63]);  view_111 = None
        constant_pad_nd_18 = torch.ops.aten.constant_pad_nd.default(view_112, [0, 1], 0.0);  view_112 = None
        view_113 = torch.ops.aten.view.default(constant_pad_nd_18, [1024, 2048]);  constant_pad_nd_18 = None
        constant_pad_nd_19 = torch.ops.aten.constant_pad_nd.default(view_113, [0, 31], 0.0);  view_113 = None
        view_114 = torch.ops.aten.view.default(constant_pad_nd_19, [-1, 33, 63]);  constant_pad_nd_19 = None
        slice_29 = torch.ops.aten.slice.Tensor(view_114, 1, 0, 32);  view_114 = None
        slice_30 = torch.ops.aten.slice.Tensor(slice_29, 2, 31, 9223372036854775807);  slice_29 = None
        view_115 = torch.ops.aten.view.default(slice_30, [32, 32, 1, 32, 32]);  slice_30 = None
        expand_27 = torch.ops.aten.expand.default(view_115, [-1, -1, 32, -1, -1]);  view_115 = None
        permute_39 = torch.ops.aten.permute.default(expand_27, [0, 3, 1, 4, 2]);  expand_27 = None
        add_134 = torch.ops.aten.add.Tensor(permute_39, permute_36);  permute_39 = permute_36 = None
        clone_34 = torch.ops.aten.clone.default(add_134, memory_format = torch.contiguous_format);  add_134 = None
        view_116 = torch.ops.aten.view.default(clone_34, [32, 1024, 1024]);  clone_34 = None
        add_135 = torch.ops.aten.add.Tensor(mul_232, view_116);  mul_232 = view_116 = None
        amax_4 = torch.ops.aten.amax.default(add_135, [-1], True)
        sub_60 = torch.ops.aten.sub.Tensor(add_135, amax_4);  add_135 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        expand_28 = torch.ops.aten.expand.default(div_4, [32, 1024, 1024]);  div_4 = None
        expand_29 = torch.ops.aten.expand.default(permute_34, [32, 1024, 32]);  permute_34 = None
        bmm_9 = torch.ops.aten.bmm.default(expand_28, expand_29);  expand_28 = expand_29 = None
        permute_40 = torch.ops.aten.permute.default(bmm_9, [0, 2, 1]);  bmm_9 = None
        clone_35 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_120 = torch.ops.aten.view.default(clone_35, [8, 128, 32, 32]);  clone_35 = None
        add_136 = torch.ops.aten.add.Tensor(arg111_1, 1e-05);  arg111_1 = None
        sqrt_56 = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_56 = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_233 = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(mul_233, -1);  mul_233 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_61 = torch.ops.aten.sub.Tensor(view_120, unsqueeze_449);  view_120 = unsqueeze_449 = None
        mul_234 = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_451);  sub_61 = unsqueeze_451 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_235 = torch.ops.aten.mul.Tensor(mul_234, unsqueeze_453);  mul_234 = unsqueeze_453 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_137 = torch.ops.aten.add.Tensor(mul_235, unsqueeze_455);  mul_235 = unsqueeze_455 = None
        sigmoid_60 = torch.ops.aten.sigmoid.default(add_137)
        mul_236 = torch.ops.aten.mul.Tensor(add_137, sigmoid_60);  add_137 = sigmoid_60 = None
        convolution_77 = torch.ops.aten.convolution.default(mul_236, arg114_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_236 = arg114_1 = None
        add_138 = torch.ops.aten.add.Tensor(arg116_1, 1e-05);  arg116_1 = None
        sqrt_57 = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_57 = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_237 = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458 = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_459 = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_62 = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_457);  convolution_77 = unsqueeze_457 = None
        mul_238 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_459);  sub_62 = unsqueeze_459 = None
        unsqueeze_460 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_461 = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_461);  mul_238 = unsqueeze_461 = None
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(arg118_1, -1);  arg118_1 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_139 = torch.ops.aten.add.Tensor(mul_239, unsqueeze_463);  mul_239 = unsqueeze_463 = None
        add_140 = torch.ops.aten.add.Tensor(add_139, mul_227);  add_139 = mul_227 = None
        sigmoid_61 = torch.ops.aten.sigmoid.default(add_140)
        mul_240 = torch.ops.aten.mul.Tensor(add_140, sigmoid_61);  add_140 = sigmoid_61 = None
        convolution_78 = torch.ops.aten.convolution.default(mul_240, arg119_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg119_1 = None
        add_141 = torch.ops.aten.add.Tensor(arg121_1, 1e-05);  arg121_1 = None
        sqrt_58 = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_58 = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_241 = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(mul_241, -1);  mul_241 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_63 = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_465);  convolution_78 = unsqueeze_465 = None
        mul_242 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_467);  sub_63 = unsqueeze_467 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_469);  mul_242 = unsqueeze_469 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_142 = torch.ops.aten.add.Tensor(mul_243, unsqueeze_471);  mul_243 = unsqueeze_471 = None
        sigmoid_62 = torch.ops.aten.sigmoid.default(add_142)
        mul_244 = torch.ops.aten.mul.Tensor(add_142, sigmoid_62);  add_142 = sigmoid_62 = None
        convolution_79 = torch.ops.aten.convolution.default(mul_244, arg124_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  mul_244 = arg124_1 = None
        add_143 = torch.ops.aten.add.Tensor(arg126_1, 1e-05);  arg126_1 = None
        sqrt_59 = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_59 = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_245 = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(mul_245, -1);  mul_245 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_64 = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_473);  convolution_79 = unsqueeze_473 = None
        mul_246 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_475);  sub_64 = unsqueeze_475 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_247 = torch.ops.aten.mul.Tensor(mul_246, unsqueeze_477);  mul_246 = unsqueeze_477 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_144 = torch.ops.aten.add.Tensor(mul_247, unsqueeze_479);  mul_247 = unsqueeze_479 = None
        sigmoid_63 = torch.ops.aten.sigmoid.default(add_144)
        mul_248 = torch.ops.aten.mul.Tensor(add_144, sigmoid_63);  add_144 = sigmoid_63 = None
        mean_11 = torch.ops.aten.mean.dim(mul_248, [2, 3], True)
        convolution_80 = torch.ops.aten.convolution.default(mean_11, arg129_1, arg130_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_11 = arg129_1 = arg130_1 = None
        relu_10 = torch.ops.aten.relu.default(convolution_80);  convolution_80 = None
        convolution_81 = torch.ops.aten.convolution.default(relu_10, arg131_1, arg132_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_10 = arg131_1 = arg132_1 = None
        sigmoid_64 = torch.ops.aten.sigmoid.default(convolution_81);  convolution_81 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, sigmoid_64);  mul_248 = sigmoid_64 = None
        convolution_82 = torch.ops.aten.convolution.default(mul_249, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_249 = arg133_1 = None
        add_145 = torch.ops.aten.add.Tensor(arg135_1, 1e-05);  arg135_1 = None
        sqrt_60 = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_60 = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_250 = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_65 = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_481);  convolution_82 = unsqueeze_481 = None
        mul_251 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_483);  sub_65 = unsqueeze_483 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_485);  mul_251 = unsqueeze_485 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_146 = torch.ops.aten.add.Tensor(mul_252, unsqueeze_487);  mul_252 = unsqueeze_487 = None
        convolution_83 = torch.ops.aten.convolution.default(mul_240, arg138_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_240 = arg138_1 = None
        add_147 = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_61 = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_61 = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_253 = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(mul_253, -1);  mul_253 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_66 = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_489);  convolution_83 = unsqueeze_489 = None
        mul_254 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_491);  sub_66 = unsqueeze_491 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_255 = torch.ops.aten.mul.Tensor(mul_254, unsqueeze_493);  mul_254 = unsqueeze_493 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_148 = torch.ops.aten.add.Tensor(mul_255, unsqueeze_495);  mul_255 = unsqueeze_495 = None
        add_149 = torch.ops.aten.add.Tensor(add_146, add_148);  add_146 = add_148 = None
        sigmoid_65 = torch.ops.aten.sigmoid.default(add_149)
        mul_256 = torch.ops.aten.mul.Tensor(add_149, sigmoid_65);  add_149 = sigmoid_65 = None
        convolution_84 = torch.ops.aten.convolution.default(mul_256, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg143_1 = None
        add_150 = torch.ops.aten.add.Tensor(arg145_1, 1e-05);  arg145_1 = None
        sqrt_62 = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_62 = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_257 = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(mul_257, -1);  mul_257 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_497);  convolution_84 = unsqueeze_497 = None
        mul_258 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_499);  sub_67 = unsqueeze_499 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_258, unsqueeze_501);  mul_258 = unsqueeze_501 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(arg147_1, -1);  arg147_1 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_151 = torch.ops.aten.add.Tensor(mul_259, unsqueeze_503);  mul_259 = unsqueeze_503 = None
        sigmoid_66 = torch.ops.aten.sigmoid.default(add_151)
        mul_260 = torch.ops.aten.mul.Tensor(add_151, sigmoid_66);  add_151 = sigmoid_66 = None
        convolution_85 = torch.ops.aten.convolution.default(mul_260, arg148_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_260 = arg148_1 = None
        add_152 = torch.ops.aten.add.Tensor(arg150_1, 1e-05);  arg150_1 = None
        sqrt_63 = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_63 = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_261 = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_68 = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_505);  convolution_85 = unsqueeze_505 = None
        mul_262 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_507);  sub_68 = unsqueeze_507 = None
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_509);  mul_262 = unsqueeze_509 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(arg152_1, -1);  arg152_1 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_153 = torch.ops.aten.add.Tensor(mul_263, unsqueeze_511);  mul_263 = unsqueeze_511 = None
        sigmoid_67 = torch.ops.aten.sigmoid.default(add_153)
        mul_264 = torch.ops.aten.mul.Tensor(add_153, sigmoid_67);  add_153 = sigmoid_67 = None
        mean_12 = torch.ops.aten.mean.dim(mul_264, [2, 3], True)
        convolution_86 = torch.ops.aten.convolution.default(mean_12, arg153_1, arg154_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_12 = arg153_1 = arg154_1 = None
        relu_11 = torch.ops.aten.relu.default(convolution_86);  convolution_86 = None
        convolution_87 = torch.ops.aten.convolution.default(relu_11, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg155_1 = arg156_1 = None
        sigmoid_68 = torch.ops.aten.sigmoid.default(convolution_87);  convolution_87 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_264, sigmoid_68);  mul_264 = sigmoid_68 = None
        convolution_88 = torch.ops.aten.convolution.default(mul_265, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_265 = arg157_1 = None
        add_154 = torch.ops.aten.add.Tensor(arg159_1, 1e-05);  arg159_1 = None
        sqrt_64 = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_64 = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_266 = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_512 = torch.ops.aten.unsqueeze.default(arg158_1, -1);  arg158_1 = None
        unsqueeze_513 = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        unsqueeze_514 = torch.ops.aten.unsqueeze.default(mul_266, -1);  mul_266 = None
        unsqueeze_515 = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        sub_69 = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_513);  convolution_88 = unsqueeze_513 = None
        mul_267 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_515);  sub_69 = unsqueeze_515 = None
        unsqueeze_516 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_517 = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_267, unsqueeze_517);  mul_267 = unsqueeze_517 = None
        unsqueeze_518 = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_519 = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_155 = torch.ops.aten.add.Tensor(mul_268, unsqueeze_519);  mul_268 = unsqueeze_519 = None
        add_156 = torch.ops.aten.add.Tensor(add_155, mul_256);  add_155 = mul_256 = None
        sigmoid_69 = torch.ops.aten.sigmoid.default(add_156)
        mul_269 = torch.ops.aten.mul.Tensor(add_156, sigmoid_69);  add_156 = sigmoid_69 = None
        convolution_89 = torch.ops.aten.convolution.default(mul_269, arg162_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg162_1 = None
        add_157 = torch.ops.aten.add.Tensor(arg164_1, 1e-05);  arg164_1 = None
        sqrt_65 = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_65 = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_270 = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_520 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_521 = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        unsqueeze_522 = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_523 = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        sub_70 = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_521);  convolution_89 = unsqueeze_521 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_523);  sub_70 = unsqueeze_523 = None
        unsqueeze_524 = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_525 = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_525);  mul_271 = unsqueeze_525 = None
        unsqueeze_526 = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_527 = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_158 = torch.ops.aten.add.Tensor(mul_272, unsqueeze_527);  mul_272 = unsqueeze_527 = None
        sigmoid_70 = torch.ops.aten.sigmoid.default(add_158)
        mul_273 = torch.ops.aten.mul.Tensor(add_158, sigmoid_70);  add_158 = sigmoid_70 = None
        convolution_90 = torch.ops.aten.convolution.default(mul_273, arg167_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_273 = arg167_1 = None
        split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(convolution_90, [256, 256, 256], 1);  convolution_90 = None
        getitem_15 = split_with_sizes_5[0]
        getitem_16 = split_with_sizes_5[1]
        getitem_17 = split_with_sizes_5[2];  split_with_sizes_5 = None
        clone_36 = torch.ops.aten.clone.default(getitem_15, memory_format = torch.contiguous_format);  getitem_15 = None
        view_121 = torch.ops.aten.view.default(clone_36, [32, 64, 256]);  clone_36 = None
        permute_41 = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
        clone_37 = torch.ops.aten.clone.default(getitem_16, memory_format = torch.contiguous_format);  getitem_16 = None
        view_122 = torch.ops.aten.view.default(clone_37, [32, 64, 256]);  clone_37 = None
        clone_38 = torch.ops.aten.clone.default(getitem_17, memory_format = torch.contiguous_format);  getitem_17 = None
        view_123 = torch.ops.aten.view.default(clone_38, [32, 64, 256]);  clone_38 = None
        permute_42 = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        expand_30 = torch.ops.aten.expand.default(permute_41, [32, 256, 64])
        expand_31 = torch.ops.aten.expand.default(view_122, [32, 64, 256]);  view_122 = None
        bmm_10 = torch.ops.aten.bmm.default(expand_30, expand_31);  expand_30 = expand_31 = None
        mul_274 = torch.ops.aten.mul.Tensor(bmm_10, 0.125);  bmm_10 = None
        view_127 = torch.ops.aten.view.default(permute_41, [32, 16, 16, 64]);  permute_41 = None
        permute_43 = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
        clone_39 = torch.ops.aten.clone.default(view_127, memory_format = torch.contiguous_format)
        view_128 = torch.ops.aten.view.default(clone_39, [8192, 64]);  clone_39 = None
        mm_10 = torch.ops.aten.mm.default(view_128, permute_43);  view_128 = permute_43 = None
        view_129 = torch.ops.aten.view.default(mm_10, [32, 16, 16, 31]);  mm_10 = None
        view_130 = torch.ops.aten.view.default(view_129, [-1, 16, 31]);  view_129 = None
        constant_pad_nd_20 = torch.ops.aten.constant_pad_nd.default(view_130, [0, 1], 0.0);  view_130 = None
        view_131 = torch.ops.aten.view.default(constant_pad_nd_20, [512, 512]);  constant_pad_nd_20 = None
        constant_pad_nd_21 = torch.ops.aten.constant_pad_nd.default(view_131, [0, 15], 0.0);  view_131 = None
        view_132 = torch.ops.aten.view.default(constant_pad_nd_21, [-1, 17, 31]);  constant_pad_nd_21 = None
        slice_32 = torch.ops.aten.slice.Tensor(view_132, 1, 0, 16);  view_132 = None
        slice_33 = torch.ops.aten.slice.Tensor(slice_32, 2, 15, 9223372036854775807);  slice_32 = None
        view_133 = torch.ops.aten.view.default(slice_33, [32, 16, 1, 16, 16]);  slice_33 = None
        expand_32 = torch.ops.aten.expand.default(view_133, [-1, -1, 16, -1, -1]);  view_133 = None
        permute_44 = torch.ops.aten.permute.default(expand_32, [0, 1, 3, 2, 4]);  expand_32 = None
        permute_45 = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
        permute_46 = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        clone_40 = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_134 = torch.ops.aten.view.default(clone_40, [8192, 64]);  clone_40 = None
        mm_11 = torch.ops.aten.mm.default(view_134, permute_46);  view_134 = permute_46 = None
        view_135 = torch.ops.aten.view.default(mm_11, [32, 16, 16, 31]);  mm_11 = None
        view_136 = torch.ops.aten.view.default(view_135, [-1, 16, 31]);  view_135 = None
        constant_pad_nd_22 = torch.ops.aten.constant_pad_nd.default(view_136, [0, 1], 0.0);  view_136 = None
        view_137 = torch.ops.aten.view.default(constant_pad_nd_22, [512, 512]);  constant_pad_nd_22 = None
        constant_pad_nd_23 = torch.ops.aten.constant_pad_nd.default(view_137, [0, 15], 0.0);  view_137 = None
        view_138 = torch.ops.aten.view.default(constant_pad_nd_23, [-1, 17, 31]);  constant_pad_nd_23 = None
        slice_35 = torch.ops.aten.slice.Tensor(view_138, 1, 0, 16);  view_138 = None
        slice_36 = torch.ops.aten.slice.Tensor(slice_35, 2, 15, 9223372036854775807);  slice_35 = None
        view_139 = torch.ops.aten.view.default(slice_36, [32, 16, 1, 16, 16]);  slice_36 = None
        expand_33 = torch.ops.aten.expand.default(view_139, [-1, -1, 16, -1, -1]);  view_139 = None
        permute_47 = torch.ops.aten.permute.default(expand_33, [0, 3, 1, 4, 2]);  expand_33 = None
        add_159 = torch.ops.aten.add.Tensor(permute_47, permute_44);  permute_47 = permute_44 = None
        clone_41 = torch.ops.aten.clone.default(add_159, memory_format = torch.contiguous_format);  add_159 = None
        view_140 = torch.ops.aten.view.default(clone_41, [32, 256, 256]);  clone_41 = None
        add_160 = torch.ops.aten.add.Tensor(mul_274, view_140);  mul_274 = view_140 = None
        amax_5 = torch.ops.aten.amax.default(add_160, [-1], True)
        sub_71 = torch.ops.aten.sub.Tensor(add_160, amax_5);  add_160 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_71);  sub_71 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        expand_34 = torch.ops.aten.expand.default(div_5, [32, 256, 256]);  div_5 = None
        expand_35 = torch.ops.aten.expand.default(permute_42, [32, 256, 64]);  permute_42 = None
        bmm_11 = torch.ops.aten.bmm.default(expand_34, expand_35);  expand_34 = expand_35 = None
        permute_48 = torch.ops.aten.permute.default(bmm_11, [0, 2, 1]);  bmm_11 = None
        clone_42 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_144 = torch.ops.aten.view.default(clone_42, [8, 256, 16, 16]);  clone_42 = None
        add_161 = torch.ops.aten.add.Tensor(arg171_1, 1e-05);  arg171_1 = None
        sqrt_66 = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_66 = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_275 = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_528 = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_529 = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        unsqueeze_530 = torch.ops.aten.unsqueeze.default(mul_275, -1);  mul_275 = None
        unsqueeze_531 = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        sub_72 = torch.ops.aten.sub.Tensor(view_144, unsqueeze_529);  view_144 = unsqueeze_529 = None
        mul_276 = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_531);  sub_72 = unsqueeze_531 = None
        unsqueeze_532 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_533 = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_277 = torch.ops.aten.mul.Tensor(mul_276, unsqueeze_533);  mul_276 = unsqueeze_533 = None
        unsqueeze_534 = torch.ops.aten.unsqueeze.default(arg173_1, -1);  arg173_1 = None
        unsqueeze_535 = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_162 = torch.ops.aten.add.Tensor(mul_277, unsqueeze_535);  mul_277 = unsqueeze_535 = None
        sigmoid_71 = torch.ops.aten.sigmoid.default(add_162)
        mul_278 = torch.ops.aten.mul.Tensor(add_162, sigmoid_71);  add_162 = sigmoid_71 = None
        convolution_91 = torch.ops.aten.convolution.default(mul_278, arg174_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_278 = arg174_1 = None
        add_163 = torch.ops.aten.add.Tensor(arg176_1, 1e-05);  arg176_1 = None
        sqrt_67 = torch.ops.aten.sqrt.default(add_163);  add_163 = None
        reciprocal_67 = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_279 = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_536 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_537 = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        unsqueeze_538 = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
        unsqueeze_539 = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        sub_73 = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_537);  convolution_91 = unsqueeze_537 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_539);  sub_73 = unsqueeze_539 = None
        unsqueeze_540 = torch.ops.aten.unsqueeze.default(arg177_1, -1);  arg177_1 = None
        unsqueeze_541 = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_541);  mul_280 = unsqueeze_541 = None
        unsqueeze_542 = torch.ops.aten.unsqueeze.default(arg178_1, -1);  arg178_1 = None
        unsqueeze_543 = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_164 = torch.ops.aten.add.Tensor(mul_281, unsqueeze_543);  mul_281 = unsqueeze_543 = None
        add_165 = torch.ops.aten.add.Tensor(add_164, mul_269);  add_164 = mul_269 = None
        sigmoid_72 = torch.ops.aten.sigmoid.default(add_165)
        mul_282 = torch.ops.aten.mul.Tensor(add_165, sigmoid_72);  add_165 = sigmoid_72 = None
        convolution_92 = torch.ops.aten.convolution.default(mul_282, arg179_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg179_1 = None
        add_166 = torch.ops.aten.add.Tensor(arg181_1, 1e-05);  arg181_1 = None
        sqrt_68 = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_68 = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_283 = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_544 = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_545 = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        unsqueeze_546 = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
        unsqueeze_547 = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        sub_74 = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_545);  convolution_92 = unsqueeze_545 = None
        mul_284 = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_547);  sub_74 = unsqueeze_547 = None
        unsqueeze_548 = torch.ops.aten.unsqueeze.default(arg182_1, -1);  arg182_1 = None
        unsqueeze_549 = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_285 = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_549);  mul_284 = unsqueeze_549 = None
        unsqueeze_550 = torch.ops.aten.unsqueeze.default(arg183_1, -1);  arg183_1 = None
        unsqueeze_551 = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_167 = torch.ops.aten.add.Tensor(mul_285, unsqueeze_551);  mul_285 = unsqueeze_551 = None
        sigmoid_73 = torch.ops.aten.sigmoid.default(add_167)
        mul_286 = torch.ops.aten.mul.Tensor(add_167, sigmoid_73);  add_167 = sigmoid_73 = None
        convolution_93 = torch.ops.aten.convolution.default(mul_286, arg184_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_286 = arg184_1 = None
        split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(convolution_93, [512, 512, 512], 1);  convolution_93 = None
        getitem_18 = split_with_sizes_6[0]
        getitem_19 = split_with_sizes_6[1]
        getitem_20 = split_with_sizes_6[2];  split_with_sizes_6 = None
        clone_43 = torch.ops.aten.clone.default(getitem_18, memory_format = torch.contiguous_format);  getitem_18 = None
        view_145 = torch.ops.aten.view.default(clone_43, [32, 128, 256]);  clone_43 = None
        permute_49 = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        clone_44 = torch.ops.aten.clone.default(getitem_19, memory_format = torch.contiguous_format);  getitem_19 = None
        view_146 = torch.ops.aten.view.default(clone_44, [32, 128, 256]);  clone_44 = None
        clone_45 = torch.ops.aten.clone.default(getitem_20, memory_format = torch.contiguous_format);  getitem_20 = None
        view_147 = torch.ops.aten.view.default(clone_45, [32, 128, 256]);  clone_45 = None
        permute_50 = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
        expand_36 = torch.ops.aten.expand.default(permute_49, [32, 256, 128])
        expand_37 = torch.ops.aten.expand.default(view_146, [32, 128, 256]);  view_146 = None
        bmm_12 = torch.ops.aten.bmm.default(expand_36, expand_37);  expand_36 = expand_37 = None
        mul_287 = torch.ops.aten.mul.Tensor(bmm_12, 0.08838834764831845);  bmm_12 = None
        view_151 = torch.ops.aten.view.default(permute_49, [32, 16, 16, 128]);  permute_49 = None
        permute_51 = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        clone_46 = torch.ops.aten.clone.default(view_151, memory_format = torch.contiguous_format)
        view_152 = torch.ops.aten.view.default(clone_46, [8192, 128]);  clone_46 = None
        mm_12 = torch.ops.aten.mm.default(view_152, permute_51);  view_152 = permute_51 = None
        view_153 = torch.ops.aten.view.default(mm_12, [32, 16, 16, 31]);  mm_12 = None
        view_154 = torch.ops.aten.view.default(view_153, [-1, 16, 31]);  view_153 = None
        constant_pad_nd_24 = torch.ops.aten.constant_pad_nd.default(view_154, [0, 1], 0.0);  view_154 = None
        view_155 = torch.ops.aten.view.default(constant_pad_nd_24, [512, 512]);  constant_pad_nd_24 = None
        constant_pad_nd_25 = torch.ops.aten.constant_pad_nd.default(view_155, [0, 15], 0.0);  view_155 = None
        view_156 = torch.ops.aten.view.default(constant_pad_nd_25, [-1, 17, 31]);  constant_pad_nd_25 = None
        slice_38 = torch.ops.aten.slice.Tensor(view_156, 1, 0, 16);  view_156 = None
        slice_39 = torch.ops.aten.slice.Tensor(slice_38, 2, 15, 9223372036854775807);  slice_38 = None
        view_157 = torch.ops.aten.view.default(slice_39, [32, 16, 1, 16, 16]);  slice_39 = None
        expand_38 = torch.ops.aten.expand.default(view_157, [-1, -1, 16, -1, -1]);  view_157 = None
        permute_52 = torch.ops.aten.permute.default(expand_38, [0, 1, 3, 2, 4]);  expand_38 = None
        permute_53 = torch.ops.aten.permute.default(view_151, [0, 2, 1, 3]);  view_151 = None
        permute_54 = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
        clone_47 = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
        view_158 = torch.ops.aten.view.default(clone_47, [8192, 128]);  clone_47 = None
        mm_13 = torch.ops.aten.mm.default(view_158, permute_54);  view_158 = permute_54 = None
        view_159 = torch.ops.aten.view.default(mm_13, [32, 16, 16, 31]);  mm_13 = None
        view_160 = torch.ops.aten.view.default(view_159, [-1, 16, 31]);  view_159 = None
        constant_pad_nd_26 = torch.ops.aten.constant_pad_nd.default(view_160, [0, 1], 0.0);  view_160 = None
        view_161 = torch.ops.aten.view.default(constant_pad_nd_26, [512, 512]);  constant_pad_nd_26 = None
        constant_pad_nd_27 = torch.ops.aten.constant_pad_nd.default(view_161, [0, 15], 0.0);  view_161 = None
        view_162 = torch.ops.aten.view.default(constant_pad_nd_27, [-1, 17, 31]);  constant_pad_nd_27 = None
        slice_41 = torch.ops.aten.slice.Tensor(view_162, 1, 0, 16);  view_162 = None
        slice_42 = torch.ops.aten.slice.Tensor(slice_41, 2, 15, 9223372036854775807);  slice_41 = None
        view_163 = torch.ops.aten.view.default(slice_42, [32, 16, 1, 16, 16]);  slice_42 = None
        expand_39 = torch.ops.aten.expand.default(view_163, [-1, -1, 16, -1, -1]);  view_163 = None
        permute_55 = torch.ops.aten.permute.default(expand_39, [0, 3, 1, 4, 2]);  expand_39 = None
        add_168 = torch.ops.aten.add.Tensor(permute_55, permute_52);  permute_55 = permute_52 = None
        clone_48 = torch.ops.aten.clone.default(add_168, memory_format = torch.contiguous_format);  add_168 = None
        view_164 = torch.ops.aten.view.default(clone_48, [32, 256, 256]);  clone_48 = None
        add_169 = torch.ops.aten.add.Tensor(mul_287, view_164);  mul_287 = view_164 = None
        amax_6 = torch.ops.aten.amax.default(add_169, [-1], True)
        sub_75 = torch.ops.aten.sub.Tensor(add_169, amax_6);  add_169 = amax_6 = None
        exp_6 = torch.ops.aten.exp.default(sub_75);  sub_75 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6 = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        expand_40 = torch.ops.aten.expand.default(div_6, [32, 256, 256]);  div_6 = None
        expand_41 = torch.ops.aten.expand.default(permute_50, [32, 256, 128]);  permute_50 = None
        bmm_13 = torch.ops.aten.bmm.default(expand_40, expand_41);  expand_40 = expand_41 = None
        permute_56 = torch.ops.aten.permute.default(bmm_13, [0, 2, 1]);  bmm_13 = None
        clone_49 = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_168 = torch.ops.aten.view.default(clone_49, [8, 512, 16, 16]);  clone_49 = None
        avg_pool2d_1 = torch.ops.aten.avg_pool2d.default(view_168, [2, 2], [2, 2]);  view_168 = None
        add_170 = torch.ops.aten.add.Tensor(arg188_1, 1e-05);  arg188_1 = None
        sqrt_69 = torch.ops.aten.sqrt.default(add_170);  add_170 = None
        reciprocal_69 = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_288 = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_552 = torch.ops.aten.unsqueeze.default(arg187_1, -1);  arg187_1 = None
        unsqueeze_553 = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        unsqueeze_554 = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_555 = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        sub_76 = torch.ops.aten.sub.Tensor(avg_pool2d_1, unsqueeze_553);  avg_pool2d_1 = unsqueeze_553 = None
        mul_289 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_555);  sub_76 = unsqueeze_555 = None
        unsqueeze_556 = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_557 = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_557);  mul_289 = unsqueeze_557 = None
        unsqueeze_558 = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_559 = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_171 = torch.ops.aten.add.Tensor(mul_290, unsqueeze_559);  mul_290 = unsqueeze_559 = None
        sigmoid_74 = torch.ops.aten.sigmoid.default(add_171)
        mul_291 = torch.ops.aten.mul.Tensor(add_171, sigmoid_74);  add_171 = sigmoid_74 = None
        convolution_94 = torch.ops.aten.convolution.default(mul_291, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_291 = arg191_1 = None
        add_172 = torch.ops.aten.add.Tensor(arg193_1, 1e-05);  arg193_1 = None
        sqrt_70 = torch.ops.aten.sqrt.default(add_172);  add_172 = None
        reciprocal_70 = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_292 = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_560 = torch.ops.aten.unsqueeze.default(arg192_1, -1);  arg192_1 = None
        unsqueeze_561 = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        unsqueeze_562 = torch.ops.aten.unsqueeze.default(mul_292, -1);  mul_292 = None
        unsqueeze_563 = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        sub_77 = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_561);  convolution_94 = unsqueeze_561 = None
        mul_293 = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_563);  sub_77 = unsqueeze_563 = None
        unsqueeze_564 = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_565 = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_294 = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_565);  mul_293 = unsqueeze_565 = None
        unsqueeze_566 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_567 = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_173 = torch.ops.aten.add.Tensor(mul_294, unsqueeze_567);  mul_294 = unsqueeze_567 = None
        convolution_95 = torch.ops.aten.convolution.default(mul_282, arg196_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_282 = arg196_1 = None
        add_174 = torch.ops.aten.add.Tensor(arg198_1, 1e-05);  arg198_1 = None
        sqrt_71 = torch.ops.aten.sqrt.default(add_174);  add_174 = None
        reciprocal_71 = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_295 = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_568 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_569 = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        unsqueeze_570 = torch.ops.aten.unsqueeze.default(mul_295, -1);  mul_295 = None
        unsqueeze_571 = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        sub_78 = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_569);  convolution_95 = unsqueeze_569 = None
        mul_296 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_571);  sub_78 = unsqueeze_571 = None
        unsqueeze_572 = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_573 = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_297 = torch.ops.aten.mul.Tensor(mul_296, unsqueeze_573);  mul_296 = unsqueeze_573 = None
        unsqueeze_574 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_575 = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_175 = torch.ops.aten.add.Tensor(mul_297, unsqueeze_575);  mul_297 = unsqueeze_575 = None
        add_176 = torch.ops.aten.add.Tensor(add_173, add_175);  add_173 = add_175 = None
        sigmoid_75 = torch.ops.aten.sigmoid.default(add_176)
        mul_298 = torch.ops.aten.mul.Tensor(add_176, sigmoid_75);  add_176 = sigmoid_75 = None
        convolution_96 = torch.ops.aten.convolution.default(mul_298, arg201_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg201_1 = None
        add_177 = torch.ops.aten.add.Tensor(arg203_1, 1e-05);  arg203_1 = None
        sqrt_72 = torch.ops.aten.sqrt.default(add_177);  add_177 = None
        reciprocal_72 = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_299 = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_576 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_577 = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        unsqueeze_578 = torch.ops.aten.unsqueeze.default(mul_299, -1);  mul_299 = None
        unsqueeze_579 = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        sub_79 = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_577);  convolution_96 = unsqueeze_577 = None
        mul_300 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_579);  sub_79 = unsqueeze_579 = None
        unsqueeze_580 = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_581 = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_301 = torch.ops.aten.mul.Tensor(mul_300, unsqueeze_581);  mul_300 = unsqueeze_581 = None
        unsqueeze_582 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_583 = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_178 = torch.ops.aten.add.Tensor(mul_301, unsqueeze_583);  mul_301 = unsqueeze_583 = None
        sigmoid_76 = torch.ops.aten.sigmoid.default(add_178)
        mul_302 = torch.ops.aten.mul.Tensor(add_178, sigmoid_76);  add_178 = sigmoid_76 = None
        convolution_97 = torch.ops.aten.convolution.default(mul_302, arg206_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_302 = arg206_1 = None
        split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(convolution_97, [512, 512, 512], 1);  convolution_97 = None
        getitem_21 = split_with_sizes_7[0]
        getitem_22 = split_with_sizes_7[1]
        getitem_23 = split_with_sizes_7[2];  split_with_sizes_7 = None
        clone_50 = torch.ops.aten.clone.default(getitem_21, memory_format = torch.contiguous_format);  getitem_21 = None
        view_169 = torch.ops.aten.view.default(clone_50, [32, 128, 64]);  clone_50 = None
        permute_57 = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        clone_51 = torch.ops.aten.clone.default(getitem_22, memory_format = torch.contiguous_format);  getitem_22 = None
        view_170 = torch.ops.aten.view.default(clone_51, [32, 128, 64]);  clone_51 = None
        clone_52 = torch.ops.aten.clone.default(getitem_23, memory_format = torch.contiguous_format);  getitem_23 = None
        view_171 = torch.ops.aten.view.default(clone_52, [32, 128, 64]);  clone_52 = None
        permute_58 = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
        expand_42 = torch.ops.aten.expand.default(permute_57, [32, 64, 128])
        expand_43 = torch.ops.aten.expand.default(view_170, [32, 128, 64]);  view_170 = None
        bmm_14 = torch.ops.aten.bmm.default(expand_42, expand_43);  expand_42 = expand_43 = None
        mul_303 = torch.ops.aten.mul.Tensor(bmm_14, 0.08838834764831845);  bmm_14 = None
        view_175 = torch.ops.aten.view.default(permute_57, [32, 8, 8, 128]);  permute_57 = None
        permute_59 = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        clone_53 = torch.ops.aten.clone.default(view_175, memory_format = torch.contiguous_format)
        view_176 = torch.ops.aten.view.default(clone_53, [2048, 128]);  clone_53 = None
        mm_14 = torch.ops.aten.mm.default(view_176, permute_59);  view_176 = permute_59 = None
        view_177 = torch.ops.aten.view.default(mm_14, [32, 8, 8, 15]);  mm_14 = None
        view_178 = torch.ops.aten.view.default(view_177, [-1, 8, 15]);  view_177 = None
        constant_pad_nd_28 = torch.ops.aten.constant_pad_nd.default(view_178, [0, 1], 0.0);  view_178 = None
        view_179 = torch.ops.aten.view.default(constant_pad_nd_28, [256, 128]);  constant_pad_nd_28 = None
        constant_pad_nd_29 = torch.ops.aten.constant_pad_nd.default(view_179, [0, 7], 0.0);  view_179 = None
        view_180 = torch.ops.aten.view.default(constant_pad_nd_29, [-1, 9, 15]);  constant_pad_nd_29 = None
        slice_44 = torch.ops.aten.slice.Tensor(view_180, 1, 0, 8);  view_180 = None
        slice_45 = torch.ops.aten.slice.Tensor(slice_44, 2, 7, 9223372036854775807);  slice_44 = None
        view_181 = torch.ops.aten.view.default(slice_45, [32, 8, 1, 8, 8]);  slice_45 = None
        expand_44 = torch.ops.aten.expand.default(view_181, [-1, -1, 8, -1, -1]);  view_181 = None
        permute_60 = torch.ops.aten.permute.default(expand_44, [0, 1, 3, 2, 4]);  expand_44 = None
        permute_61 = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        permute_62 = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        clone_54 = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_182 = torch.ops.aten.view.default(clone_54, [2048, 128]);  clone_54 = None
        mm_15 = torch.ops.aten.mm.default(view_182, permute_62);  view_182 = permute_62 = None
        view_183 = torch.ops.aten.view.default(mm_15, [32, 8, 8, 15]);  mm_15 = None
        view_184 = torch.ops.aten.view.default(view_183, [-1, 8, 15]);  view_183 = None
        constant_pad_nd_30 = torch.ops.aten.constant_pad_nd.default(view_184, [0, 1], 0.0);  view_184 = None
        view_185 = torch.ops.aten.view.default(constant_pad_nd_30, [256, 128]);  constant_pad_nd_30 = None
        constant_pad_nd_31 = torch.ops.aten.constant_pad_nd.default(view_185, [0, 7], 0.0);  view_185 = None
        view_186 = torch.ops.aten.view.default(constant_pad_nd_31, [-1, 9, 15]);  constant_pad_nd_31 = None
        slice_47 = torch.ops.aten.slice.Tensor(view_186, 1, 0, 8);  view_186 = None
        slice_48 = torch.ops.aten.slice.Tensor(slice_47, 2, 7, 9223372036854775807);  slice_47 = None
        view_187 = torch.ops.aten.view.default(slice_48, [32, 8, 1, 8, 8]);  slice_48 = None
        expand_45 = torch.ops.aten.expand.default(view_187, [-1, -1, 8, -1, -1]);  view_187 = None
        permute_63 = torch.ops.aten.permute.default(expand_45, [0, 3, 1, 4, 2]);  expand_45 = None
        add_179 = torch.ops.aten.add.Tensor(permute_63, permute_60);  permute_63 = permute_60 = None
        clone_55 = torch.ops.aten.clone.default(add_179, memory_format = torch.contiguous_format);  add_179 = None
        view_188 = torch.ops.aten.view.default(clone_55, [32, 64, 64]);  clone_55 = None
        add_180 = torch.ops.aten.add.Tensor(mul_303, view_188);  mul_303 = view_188 = None
        amax_7 = torch.ops.aten.amax.default(add_180, [-1], True)
        sub_80 = torch.ops.aten.sub.Tensor(add_180, amax_7);  add_180 = amax_7 = None
        exp_7 = torch.ops.aten.exp.default(sub_80);  sub_80 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7 = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        expand_46 = torch.ops.aten.expand.default(div_7, [32, 64, 64]);  div_7 = None
        expand_47 = torch.ops.aten.expand.default(permute_58, [32, 64, 128]);  permute_58 = None
        bmm_15 = torch.ops.aten.bmm.default(expand_46, expand_47);  expand_46 = expand_47 = None
        permute_64 = torch.ops.aten.permute.default(bmm_15, [0, 2, 1]);  bmm_15 = None
        clone_56 = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_192 = torch.ops.aten.view.default(clone_56, [8, 512, 8, 8]);  clone_56 = None
        add_181 = torch.ops.aten.add.Tensor(arg210_1, 1e-05);  arg210_1 = None
        sqrt_73 = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_73 = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_304 = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_584 = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_585 = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        unsqueeze_586 = torch.ops.aten.unsqueeze.default(mul_304, -1);  mul_304 = None
        unsqueeze_587 = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        sub_81 = torch.ops.aten.sub.Tensor(view_192, unsqueeze_585);  view_192 = unsqueeze_585 = None
        mul_305 = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_587);  sub_81 = unsqueeze_587 = None
        unsqueeze_588 = torch.ops.aten.unsqueeze.default(arg211_1, -1);  arg211_1 = None
        unsqueeze_589 = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_306 = torch.ops.aten.mul.Tensor(mul_305, unsqueeze_589);  mul_305 = unsqueeze_589 = None
        unsqueeze_590 = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_591 = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_182 = torch.ops.aten.add.Tensor(mul_306, unsqueeze_591);  mul_306 = unsqueeze_591 = None
        sigmoid_77 = torch.ops.aten.sigmoid.default(add_182)
        mul_307 = torch.ops.aten.mul.Tensor(add_182, sigmoid_77);  add_182 = sigmoid_77 = None
        convolution_98 = torch.ops.aten.convolution.default(mul_307, arg213_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_307 = arg213_1 = None
        add_183 = torch.ops.aten.add.Tensor(arg215_1, 1e-05);  arg215_1 = None
        sqrt_74 = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_74 = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_308 = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_592 = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_593 = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        unsqueeze_594 = torch.ops.aten.unsqueeze.default(mul_308, -1);  mul_308 = None
        unsqueeze_595 = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        sub_82 = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_593);  convolution_98 = unsqueeze_593 = None
        mul_309 = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_595);  sub_82 = unsqueeze_595 = None
        unsqueeze_596 = torch.ops.aten.unsqueeze.default(arg216_1, -1);  arg216_1 = None
        unsqueeze_597 = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_310 = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_597);  mul_309 = unsqueeze_597 = None
        unsqueeze_598 = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_599 = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_184 = torch.ops.aten.add.Tensor(mul_310, unsqueeze_599);  mul_310 = unsqueeze_599 = None
        add_185 = torch.ops.aten.add.Tensor(add_184, mul_298);  add_184 = mul_298 = None
        sigmoid_78 = torch.ops.aten.sigmoid.default(add_185)
        mul_311 = torch.ops.aten.mul.Tensor(add_185, sigmoid_78);  add_185 = sigmoid_78 = None
        convolution_99 = torch.ops.aten.convolution.default(mul_311, arg218_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_311 = arg218_1 = None
        add_186 = torch.ops.aten.add.Tensor(arg220_1, 1e-05);  arg220_1 = None
        sqrt_75 = torch.ops.aten.sqrt.default(add_186);  add_186 = None
        reciprocal_75 = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_312 = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_600 = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_601 = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        unsqueeze_602 = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
        unsqueeze_603 = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        sub_83 = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_601);  convolution_99 = unsqueeze_601 = None
        mul_313 = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_603);  sub_83 = unsqueeze_603 = None
        unsqueeze_604 = torch.ops.aten.unsqueeze.default(arg221_1, -1);  arg221_1 = None
        unsqueeze_605 = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_605);  mul_313 = unsqueeze_605 = None
        unsqueeze_606 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_607 = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_187 = torch.ops.aten.add.Tensor(mul_314, unsqueeze_607);  mul_314 = unsqueeze_607 = None
        sigmoid_79 = torch.ops.aten.sigmoid.default(add_187)
        mul_315 = torch.ops.aten.mul.Tensor(add_187, sigmoid_79);  add_187 = sigmoid_79 = None
        mean_13 = torch.ops.aten.mean.dim(mul_315, [-1, -2], True);  mul_315 = None
        view_193 = torch.ops.aten.view.default(mean_13, [8, 1280]);  mean_13 = None
        permute_65 = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg224_1, view_193, permute_65);  arg224_1 = view_193 = permute_65 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 2592, device=device(type='cuda', index=0))
    reader.tensor(buf0, (24, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 256, 256), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf2, (24,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf3, (24,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf4, (24,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 96, device=device(type='cuda', index=0))
    reader.tensor(buf5, (24,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 27648, device=device(type='cuda', index=0))
    reader.tensor(buf6, (32, 24, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf7, (32,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf8, (32,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf9, (32,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf10, (32,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64, 32, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf16, (64, 64, 1, 1), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf17, (64,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf18, (64,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf19, (64,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf20, (64,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf21, (64, 64, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf22, (64,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf23, (64,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf25, (64,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf26, (8, 64, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf27, (8,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf28, (64, 8, 1, 1), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf29, (64,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256, 64, 1, 1), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (256,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256, 64, 1, 1), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf40, (64, 256, 1, 1), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf41, (64,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf42, (64,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf43, (64,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64, 64, 3, 3), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (8, 64, 1, 1), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf51, (8,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf52, (64, 8, 1, 1), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf53, (64,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf54, (256, 64, 1, 1), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf59, (128, 256, 1, 1), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf60, (128,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128, 128, 3, 3), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf65, (128,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf66, (128,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf67, (128,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf68, (128,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf69, (8, 128, 1, 1), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf70, (8,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf71, (128, 8, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf72, (128,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512, 128, 1, 1), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf74, (512,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf75, (512,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf76, (512,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf77, (512,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf78, (512, 256, 1, 1), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf79, (512,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf80, (512,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf81, (512,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf82, (512,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128, 512, 1, 1), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf85, (128,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf86, (128,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf87, (128,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf88, (128, 128, 3, 3), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf89, (128,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf90, (128,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf91, (128,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf92, (128,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf93, (8, 128, 1, 1), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf94, (8,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf95, (128, 8, 1, 1), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf96, (128,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf97, (512, 128, 1, 1), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf98, (512,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf99, (512,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf100, (512,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf101, (512,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf102, (128, 512, 1, 1), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf103, (128,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf104, (128,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf105, (128,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf106, (128,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf107, (384, 128, 1, 1), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 8064, device=device(type='cuda', index=0))
    reader.tensor(buf108, (63, 32), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 8064, device=device(type='cuda', index=0))
    reader.tensor(buf109, (63, 32), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf110, (128,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf111, (128,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf112, (128,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf113, (128,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf114, (512, 128, 1, 1), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf115, (512,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf116, (512,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf117, (512,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf118, (512,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf119, (256, 512, 1, 1), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf120, (256,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf122, (256,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf124, (256, 256, 3, 3), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf125, (256,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf126, (256,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf127, (256,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf128, (256,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf129, (16, 256, 1, 1), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf130, (16,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf131, (256, 16, 1, 1), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf132, (256,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1024, 256, 1, 1), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf134, (1024,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1024,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf136, (1024,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf137, (1024,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf138, (1024, 512, 1, 1), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1024,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf140, (1024,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf141, (1024,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf142, (1024,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf143, (256, 1024, 1, 1), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf144, (256,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf145, (256,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf146, (256,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf147, (256,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf148, (256, 256, 3, 3), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf149, (256,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf150, (256,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf151, (256,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf152, (256,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf153, (16, 256, 1, 1), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf154, (16,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf155, (256, 16, 1, 1), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf156, (256,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf157, (1024, 256, 1, 1), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf158, (1024,), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf159, (1024,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf160, (1024,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf161, (1024,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf162, (256, 1024, 1, 1), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf163, (256,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf164, (256,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf165, (256,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf166, (256,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 786432, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768, 256, 1, 1), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 7936, device=device(type='cuda', index=0))
    reader.tensor(buf168, (31, 64), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 7936, device=device(type='cuda', index=0))
    reader.tensor(buf169, (31, 64), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf170, (256,), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf171, (256,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf172, (256,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf173, (256,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf174, (1024, 256, 1, 1), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf175, (1024,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf176, (1024,), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf177, (1024,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf178, (1024,), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf179, (512, 1024, 1, 1), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf180, (512,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf181, (512,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf182, (512,), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf183, (512,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf184, (1536, 512, 1, 1), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 15872, device=device(type='cuda', index=0))
    reader.tensor(buf185, (31, 128), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 15872, device=device(type='cuda', index=0))
    reader.tensor(buf186, (31, 128), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf187, (512,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf188, (512,), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf189, (512,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf190, (512,), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf191, (1536, 512, 1, 1), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf192, (1536,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf193, (1536,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf194, (1536,), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf195, (1536,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf196, (1536, 1024, 1, 1), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf197, (1536,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf198, (1536,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf199, (1536,), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf200, (1536,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf201, (512, 1536, 1, 1), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf202, (512,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf203, (512,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf204, (512,), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf205, (512,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf206, (1536, 512, 1, 1), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf207, (15, 128), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 7680, device=device(type='cuda', index=0))
    reader.tensor(buf208, (15, 128), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf209, (512,), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf210, (512,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf211, (512,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf212, (512,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf213, (1536, 512, 1, 1), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf214, (1536,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf215, (1536,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf216, (1536,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf217, (1536,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 7864320, device=device(type='cuda', index=0))
    reader.tensor(buf218, (1280, 1536, 1, 1), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf219, (1280,), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf220, (1280,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf221, (1280,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf222, (1280,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 5120000, device=device(type='cuda', index=0))
    reader.tensor(buf223, (1000, 1280), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf224, (1000,), is_leaf=True)  # arg224_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)