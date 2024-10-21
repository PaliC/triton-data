
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1):
        constant_pad_nd_5 = torch.ops.aten.constant_pad_nd.default(arg0_1, [0, 1, 0, 1], 0.0);  arg0_1 = None
        view_172 = torch.ops.aten.view.default(arg1_1, [1, 16, -1]);  arg1_1 = None
        mul_439 = torch.ops.aten.mul.Tensor(arg2_1, 0.19245008972987526);  arg2_1 = None
        view_173 = torch.ops.aten.view.default(mul_439, [-1]);  mul_439 = None
        var_mean_57 = torch.ops.aten.var_mean.correction(view_172, [0, 2], correction = 0, keepdim = True)
        getitem_114 = var_mean_57[0]
        getitem_115 = var_mean_57[1];  var_mean_57 = None
        add_121 = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
        rsqrt_57 = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        sub_57 = torch.ops.aten.sub.Tensor(view_172, getitem_115);  view_172 = getitem_115 = None
        mul_440 = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(view_173, -1);  view_173 = None
        mul_441 = torch.ops.aten.mul.Tensor(mul_440, unsqueeze_57);  mul_440 = unsqueeze_57 = None
        view_174 = torch.ops.aten.view.default(mul_441, [16, 3, 3, 3]);  mul_441 = None
        convolution_81 = torch.ops.aten.convolution.default(constant_pad_nd_5, view_174, arg3_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  constant_pad_nd_5 = view_174 = arg3_1 = None
        mul_442 = torch.ops.aten.mul.Tensor(convolution_81, 0.5)
        mul_443 = torch.ops.aten.mul.Tensor(convolution_81, 0.7071067811865476);  convolution_81 = None
        erf_52 = torch.ops.aten.erf.default(mul_443);  mul_443 = None
        add_122 = torch.ops.aten.add.Tensor(erf_52, 1);  erf_52 = None
        mul_444 = torch.ops.aten.mul.Tensor(mul_442, add_122);  mul_442 = add_122 = None
        mul_445 = torch.ops.aten.mul.Tensor(mul_444, 1.7015043497085571);  mul_444 = None
        view_175 = torch.ops.aten.view.default(arg4_1, [1, 32, -1]);  arg4_1 = None
        mul_446 = torch.ops.aten.mul.Tensor(arg5_1, 0.08333333333333333);  arg5_1 = None
        view_176 = torch.ops.aten.view.default(mul_446, [-1]);  mul_446 = None
        var_mean_58 = torch.ops.aten.var_mean.correction(view_175, [0, 2], correction = 0, keepdim = True)
        getitem_116 = var_mean_58[0]
        getitem_117 = var_mean_58[1];  var_mean_58 = None
        add_123 = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_58 = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_58 = torch.ops.aten.sub.Tensor(view_175, getitem_117);  view_175 = getitem_117 = None
        mul_447 = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(view_176, -1);  view_176 = None
        mul_448 = torch.ops.aten.mul.Tensor(mul_447, unsqueeze_58);  mul_447 = unsqueeze_58 = None
        view_177 = torch.ops.aten.view.default(mul_448, [32, 16, 3, 3]);  mul_448 = None
        convolution_82 = torch.ops.aten.convolution.default(mul_445, view_177, arg6_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_445 = view_177 = arg6_1 = None
        mul_449 = torch.ops.aten.mul.Tensor(convolution_82, 0.5)
        mul_450 = torch.ops.aten.mul.Tensor(convolution_82, 0.7071067811865476);  convolution_82 = None
        erf_53 = torch.ops.aten.erf.default(mul_450);  mul_450 = None
        add_124 = torch.ops.aten.add.Tensor(erf_53, 1);  erf_53 = None
        mul_451 = torch.ops.aten.mul.Tensor(mul_449, add_124);  mul_449 = add_124 = None
        mul_452 = torch.ops.aten.mul.Tensor(mul_451, 1.7015043497085571);  mul_451 = None
        view_178 = torch.ops.aten.view.default(arg7_1, [1, 64, -1]);  arg7_1 = None
        mul_453 = torch.ops.aten.mul.Tensor(arg8_1, 0.05892556509887896);  arg8_1 = None
        view_179 = torch.ops.aten.view.default(mul_453, [-1]);  mul_453 = None
        var_mean_59 = torch.ops.aten.var_mean.correction(view_178, [0, 2], correction = 0, keepdim = True)
        getitem_118 = var_mean_59[0]
        getitem_119 = var_mean_59[1];  var_mean_59 = None
        add_125 = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_59 = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
        sub_59 = torch.ops.aten.sub.Tensor(view_178, getitem_119);  view_178 = getitem_119 = None
        mul_454 = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(view_179, -1);  view_179 = None
        mul_455 = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_59);  mul_454 = unsqueeze_59 = None
        view_180 = torch.ops.aten.view.default(mul_455, [64, 32, 3, 3]);  mul_455 = None
        convolution_83 = torch.ops.aten.convolution.default(mul_452, view_180, arg9_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_452 = view_180 = arg9_1 = None
        mul_456 = torch.ops.aten.mul.Tensor(convolution_83, 0.5)
        mul_457 = torch.ops.aten.mul.Tensor(convolution_83, 0.7071067811865476);  convolution_83 = None
        erf_54 = torch.ops.aten.erf.default(mul_457);  mul_457 = None
        add_126 = torch.ops.aten.add.Tensor(erf_54, 1);  erf_54 = None
        mul_458 = torch.ops.aten.mul.Tensor(mul_456, add_126);  mul_456 = add_126 = None
        mul_459 = torch.ops.aten.mul.Tensor(mul_458, 1.7015043497085571);  mul_458 = None
        constant_pad_nd_6 = torch.ops.aten.constant_pad_nd.default(mul_459, [0, 1, 0, 1], 0.0);  mul_459 = None
        view_181 = torch.ops.aten.view.default(arg10_1, [1, 128, -1]);  arg10_1 = None
        mul_460 = torch.ops.aten.mul.Tensor(arg11_1, 0.041666666666666664);  arg11_1 = None
        view_182 = torch.ops.aten.view.default(mul_460, [-1]);  mul_460 = None
        var_mean_60 = torch.ops.aten.var_mean.correction(view_181, [0, 2], correction = 0, keepdim = True)
        getitem_120 = var_mean_60[0]
        getitem_121 = var_mean_60[1];  var_mean_60 = None
        add_127 = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
        rsqrt_60 = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_60 = torch.ops.aten.sub.Tensor(view_181, getitem_121);  view_181 = getitem_121 = None
        mul_461 = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(view_182, -1);  view_182 = None
        mul_462 = torch.ops.aten.mul.Tensor(mul_461, unsqueeze_60);  mul_461 = unsqueeze_60 = None
        view_183 = torch.ops.aten.view.default(mul_462, [128, 64, 3, 3]);  mul_462 = None
        convolution_84 = torch.ops.aten.convolution.default(constant_pad_nd_6, view_183, arg12_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  constant_pad_nd_6 = view_183 = arg12_1 = None
        mul_463 = torch.ops.aten.mul.Tensor(convolution_84, 0.5)
        mul_464 = torch.ops.aten.mul.Tensor(convolution_84, 0.7071067811865476);  convolution_84 = None
        erf_55 = torch.ops.aten.erf.default(mul_464);  mul_464 = None
        add_128 = torch.ops.aten.add.Tensor(erf_55, 1);  erf_55 = None
        mul_465 = torch.ops.aten.mul.Tensor(mul_463, add_128);  mul_463 = add_128 = None
        mul_466 = torch.ops.aten.mul.Tensor(mul_465, 1.7015043497085571);  mul_465 = None
        mul_467 = torch.ops.aten.mul.Tensor(mul_466, 1.0);  mul_466 = None
        view_184 = torch.ops.aten.view.default(arg13_1, [1, 256, -1]);  arg13_1 = None
        mul_468 = torch.ops.aten.mul.Tensor(arg14_1, 0.08838834764831845);  arg14_1 = None
        view_185 = torch.ops.aten.view.default(mul_468, [-1]);  mul_468 = None
        var_mean_61 = torch.ops.aten.var_mean.correction(view_184, [0, 2], correction = 0, keepdim = True)
        getitem_122 = var_mean_61[0]
        getitem_123 = var_mean_61[1];  var_mean_61 = None
        add_129 = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
        rsqrt_61 = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        sub_61 = torch.ops.aten.sub.Tensor(view_184, getitem_123);  view_184 = getitem_123 = None
        mul_469 = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(view_185, -1);  view_185 = None
        mul_470 = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_61);  mul_469 = unsqueeze_61 = None
        view_186 = torch.ops.aten.view.default(mul_470, [256, 128, 1, 1]);  mul_470 = None
        convolution_85 = torch.ops.aten.convolution.default(mul_467, view_186, arg15_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_186 = arg15_1 = None
        view_187 = torch.ops.aten.view.default(arg16_1, [1, 128, -1]);  arg16_1 = None
        mul_471 = torch.ops.aten.mul.Tensor(arg17_1, 0.08838834764831845);  arg17_1 = None
        view_188 = torch.ops.aten.view.default(mul_471, [-1]);  mul_471 = None
        var_mean_62 = torch.ops.aten.var_mean.correction(view_187, [0, 2], correction = 0, keepdim = True)
        getitem_124 = var_mean_62[0]
        getitem_125 = var_mean_62[1];  var_mean_62 = None
        add_130 = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
        rsqrt_62 = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_62 = torch.ops.aten.sub.Tensor(view_187, getitem_125);  view_187 = getitem_125 = None
        mul_472 = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(view_188, -1);  view_188 = None
        mul_473 = torch.ops.aten.mul.Tensor(mul_472, unsqueeze_62);  mul_472 = unsqueeze_62 = None
        view_189 = torch.ops.aten.view.default(mul_473, [128, 128, 1, 1]);  mul_473 = None
        convolution_86 = torch.ops.aten.convolution.default(mul_467, view_189, arg18_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_467 = view_189 = arg18_1 = None
        mul_474 = torch.ops.aten.mul.Tensor(convolution_86, 0.5)
        mul_475 = torch.ops.aten.mul.Tensor(convolution_86, 0.7071067811865476);  convolution_86 = None
        erf_56 = torch.ops.aten.erf.default(mul_475);  mul_475 = None
        add_131 = torch.ops.aten.add.Tensor(erf_56, 1);  erf_56 = None
        mul_476 = torch.ops.aten.mul.Tensor(mul_474, add_131);  mul_474 = add_131 = None
        mul_477 = torch.ops.aten.mul.Tensor(mul_476, 1.7015043497085571);  mul_476 = None
        view_190 = torch.ops.aten.view.default(arg19_1, [1, 128, -1]);  arg19_1 = None
        mul_478 = torch.ops.aten.mul.Tensor(arg20_1, 0.02946278254943948);  arg20_1 = None
        view_191 = torch.ops.aten.view.default(mul_478, [-1]);  mul_478 = None
        var_mean_63 = torch.ops.aten.var_mean.correction(view_190, [0, 2], correction = 0, keepdim = True)
        getitem_126 = var_mean_63[0]
        getitem_127 = var_mean_63[1];  var_mean_63 = None
        add_132 = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_63 = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        sub_63 = torch.ops.aten.sub.Tensor(view_190, getitem_127);  view_190 = getitem_127 = None
        mul_479 = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(view_191, -1);  view_191 = None
        mul_480 = torch.ops.aten.mul.Tensor(mul_479, unsqueeze_63);  mul_479 = unsqueeze_63 = None
        view_192 = torch.ops.aten.view.default(mul_480, [128, 128, 3, 3]);  mul_480 = None
        convolution_87 = torch.ops.aten.convolution.default(mul_477, view_192, arg21_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_477 = view_192 = arg21_1 = None
        mul_481 = torch.ops.aten.mul.Tensor(convolution_87, 0.5)
        mul_482 = torch.ops.aten.mul.Tensor(convolution_87, 0.7071067811865476);  convolution_87 = None
        erf_57 = torch.ops.aten.erf.default(mul_482);  mul_482 = None
        add_133 = torch.ops.aten.add.Tensor(erf_57, 1);  erf_57 = None
        mul_483 = torch.ops.aten.mul.Tensor(mul_481, add_133);  mul_481 = add_133 = None
        mul_484 = torch.ops.aten.mul.Tensor(mul_483, 1.7015043497085571);  mul_483 = None
        view_193 = torch.ops.aten.view.default(arg22_1, [1, 128, -1]);  arg22_1 = None
        mul_485 = torch.ops.aten.mul.Tensor(arg23_1, 0.02946278254943948);  arg23_1 = None
        view_194 = torch.ops.aten.view.default(mul_485, [-1]);  mul_485 = None
        var_mean_64 = torch.ops.aten.var_mean.correction(view_193, [0, 2], correction = 0, keepdim = True)
        getitem_128 = var_mean_64[0]
        getitem_129 = var_mean_64[1];  var_mean_64 = None
        add_134 = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_64 = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_64 = torch.ops.aten.sub.Tensor(view_193, getitem_129);  view_193 = getitem_129 = None
        mul_486 = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(view_194, -1);  view_194 = None
        mul_487 = torch.ops.aten.mul.Tensor(mul_486, unsqueeze_64);  mul_486 = unsqueeze_64 = None
        view_195 = torch.ops.aten.view.default(mul_487, [128, 128, 3, 3]);  mul_487 = None
        convolution_88 = torch.ops.aten.convolution.default(mul_484, view_195, arg24_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_484 = view_195 = arg24_1 = None
        mul_488 = torch.ops.aten.mul.Tensor(convolution_88, 0.5)
        mul_489 = torch.ops.aten.mul.Tensor(convolution_88, 0.7071067811865476);  convolution_88 = None
        erf_58 = torch.ops.aten.erf.default(mul_489);  mul_489 = None
        add_135 = torch.ops.aten.add.Tensor(erf_58, 1);  erf_58 = None
        mul_490 = torch.ops.aten.mul.Tensor(mul_488, add_135);  mul_488 = add_135 = None
        mul_491 = torch.ops.aten.mul.Tensor(mul_490, 1.7015043497085571);  mul_490 = None
        view_196 = torch.ops.aten.view.default(arg25_1, [1, 256, -1]);  arg25_1 = None
        mul_492 = torch.ops.aten.mul.Tensor(arg26_1, 0.08838834764831845);  arg26_1 = None
        view_197 = torch.ops.aten.view.default(mul_492, [-1]);  mul_492 = None
        var_mean_65 = torch.ops.aten.var_mean.correction(view_196, [0, 2], correction = 0, keepdim = True)
        getitem_130 = var_mean_65[0]
        getitem_131 = var_mean_65[1];  var_mean_65 = None
        add_136 = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_65 = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_65 = torch.ops.aten.sub.Tensor(view_196, getitem_131);  view_196 = getitem_131 = None
        mul_493 = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(view_197, -1);  view_197 = None
        mul_494 = torch.ops.aten.mul.Tensor(mul_493, unsqueeze_65);  mul_493 = unsqueeze_65 = None
        view_198 = torch.ops.aten.view.default(mul_494, [256, 128, 1, 1]);  mul_494 = None
        convolution_89 = torch.ops.aten.convolution.default(mul_491, view_198, arg27_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_491 = view_198 = arg27_1 = None
        mean_13 = torch.ops.aten.mean.dim(convolution_89, [2, 3], True)
        convolution_90 = torch.ops.aten.convolution.default(mean_13, arg28_1, arg29_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_13 = arg28_1 = arg29_1 = None
        relu_12 = torch.ops.aten.relu.default(convolution_90);  convolution_90 = None
        convolution_91 = torch.ops.aten.convolution.default(relu_12, arg30_1, arg31_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_12 = arg30_1 = arg31_1 = None
        sigmoid_12 = torch.ops.aten.sigmoid.default(convolution_91);  convolution_91 = None
        mul_495 = torch.ops.aten.mul.Tensor(convolution_89, sigmoid_12);  convolution_89 = sigmoid_12 = None
        mul_496 = torch.ops.aten.mul.Tensor(mul_495, 2.0);  mul_495 = None
        mul_497 = torch.ops.aten.mul.Tensor(mul_496, arg32_1);  mul_496 = arg32_1 = None
        mul_498 = torch.ops.aten.mul.Tensor(mul_497, 0.2);  mul_497 = None
        add_137 = torch.ops.aten.add.Tensor(mul_498, convolution_85);  mul_498 = convolution_85 = None
        mul_499 = torch.ops.aten.mul.Tensor(add_137, 0.5)
        mul_500 = torch.ops.aten.mul.Tensor(add_137, 0.7071067811865476);  add_137 = None
        erf_59 = torch.ops.aten.erf.default(mul_500);  mul_500 = None
        add_138 = torch.ops.aten.add.Tensor(erf_59, 1);  erf_59 = None
        mul_501 = torch.ops.aten.mul.Tensor(mul_499, add_138);  mul_499 = add_138 = None
        mul_502 = torch.ops.aten.mul.Tensor(mul_501, 1.7015043497085571);  mul_501 = None
        mul_503 = torch.ops.aten.mul.Tensor(mul_502, 0.9805806756909201);  mul_502 = None
        avg_pool2d_3 = torch.ops.aten.avg_pool2d.default(mul_503, [2, 2], [2, 2], [0, 0], True, False)
        view_199 = torch.ops.aten.view.default(arg33_1, [1, 512, -1]);  arg33_1 = None
        mul_504 = torch.ops.aten.mul.Tensor(arg34_1, 0.0625);  arg34_1 = None
        view_200 = torch.ops.aten.view.default(mul_504, [-1]);  mul_504 = None
        var_mean_66 = torch.ops.aten.var_mean.correction(view_199, [0, 2], correction = 0, keepdim = True)
        getitem_132 = var_mean_66[0]
        getitem_133 = var_mean_66[1];  var_mean_66 = None
        add_139 = torch.ops.aten.add.Tensor(getitem_132, 1e-05);  getitem_132 = None
        rsqrt_66 = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
        sub_66 = torch.ops.aten.sub.Tensor(view_199, getitem_133);  view_199 = getitem_133 = None
        mul_505 = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(view_200, -1);  view_200 = None
        mul_506 = torch.ops.aten.mul.Tensor(mul_505, unsqueeze_66);  mul_505 = unsqueeze_66 = None
        view_201 = torch.ops.aten.view.default(mul_506, [512, 256, 1, 1]);  mul_506 = None
        convolution_92 = torch.ops.aten.convolution.default(avg_pool2d_3, view_201, arg35_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_3 = view_201 = arg35_1 = None
        view_202 = torch.ops.aten.view.default(arg36_1, [1, 256, -1]);  arg36_1 = None
        mul_507 = torch.ops.aten.mul.Tensor(arg37_1, 0.0625);  arg37_1 = None
        view_203 = torch.ops.aten.view.default(mul_507, [-1]);  mul_507 = None
        var_mean_67 = torch.ops.aten.var_mean.correction(view_202, [0, 2], correction = 0, keepdim = True)
        getitem_134 = var_mean_67[0]
        getitem_135 = var_mean_67[1];  var_mean_67 = None
        add_140 = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
        rsqrt_67 = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        sub_67 = torch.ops.aten.sub.Tensor(view_202, getitem_135);  view_202 = getitem_135 = None
        mul_508 = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(view_203, -1);  view_203 = None
        mul_509 = torch.ops.aten.mul.Tensor(mul_508, unsqueeze_67);  mul_508 = unsqueeze_67 = None
        view_204 = torch.ops.aten.view.default(mul_509, [256, 256, 1, 1]);  mul_509 = None
        convolution_93 = torch.ops.aten.convolution.default(mul_503, view_204, arg38_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_503 = view_204 = arg38_1 = None
        mul_510 = torch.ops.aten.mul.Tensor(convolution_93, 0.5)
        mul_511 = torch.ops.aten.mul.Tensor(convolution_93, 0.7071067811865476);  convolution_93 = None
        erf_60 = torch.ops.aten.erf.default(mul_511);  mul_511 = None
        add_141 = torch.ops.aten.add.Tensor(erf_60, 1);  erf_60 = None
        mul_512 = torch.ops.aten.mul.Tensor(mul_510, add_141);  mul_510 = add_141 = None
        mul_513 = torch.ops.aten.mul.Tensor(mul_512, 1.7015043497085571);  mul_512 = None
        constant_pad_nd_7 = torch.ops.aten.constant_pad_nd.default(mul_513, [0, 1, 0, 1], 0.0);  mul_513 = None
        view_205 = torch.ops.aten.view.default(arg39_1, [1, 256, -1]);  arg39_1 = None
        mul_514 = torch.ops.aten.mul.Tensor(arg40_1, 0.02946278254943948);  arg40_1 = None
        view_206 = torch.ops.aten.view.default(mul_514, [-1]);  mul_514 = None
        var_mean_68 = torch.ops.aten.var_mean.correction(view_205, [0, 2], correction = 0, keepdim = True)
        getitem_136 = var_mean_68[0]
        getitem_137 = var_mean_68[1];  var_mean_68 = None
        add_142 = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
        rsqrt_68 = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_68 = torch.ops.aten.sub.Tensor(view_205, getitem_137);  view_205 = getitem_137 = None
        mul_515 = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(view_206, -1);  view_206 = None
        mul_516 = torch.ops.aten.mul.Tensor(mul_515, unsqueeze_68);  mul_515 = unsqueeze_68 = None
        view_207 = torch.ops.aten.view.default(mul_516, [256, 128, 3, 3]);  mul_516 = None
        convolution_94 = torch.ops.aten.convolution.default(constant_pad_nd_7, view_207, arg41_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 2);  constant_pad_nd_7 = view_207 = arg41_1 = None
        mul_517 = torch.ops.aten.mul.Tensor(convolution_94, 0.5)
        mul_518 = torch.ops.aten.mul.Tensor(convolution_94, 0.7071067811865476);  convolution_94 = None
        erf_61 = torch.ops.aten.erf.default(mul_518);  mul_518 = None
        add_143 = torch.ops.aten.add.Tensor(erf_61, 1);  erf_61 = None
        mul_519 = torch.ops.aten.mul.Tensor(mul_517, add_143);  mul_517 = add_143 = None
        mul_520 = torch.ops.aten.mul.Tensor(mul_519, 1.7015043497085571);  mul_519 = None
        view_208 = torch.ops.aten.view.default(arg42_1, [1, 256, -1]);  arg42_1 = None
        mul_521 = torch.ops.aten.mul.Tensor(arg43_1, 0.02946278254943948);  arg43_1 = None
        view_209 = torch.ops.aten.view.default(mul_521, [-1]);  mul_521 = None
        var_mean_69 = torch.ops.aten.var_mean.correction(view_208, [0, 2], correction = 0, keepdim = True)
        getitem_138 = var_mean_69[0]
        getitem_139 = var_mean_69[1];  var_mean_69 = None
        add_144 = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
        rsqrt_69 = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_69 = torch.ops.aten.sub.Tensor(view_208, getitem_139);  view_208 = getitem_139 = None
        mul_522 = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(view_209, -1);  view_209 = None
        mul_523 = torch.ops.aten.mul.Tensor(mul_522, unsqueeze_69);  mul_522 = unsqueeze_69 = None
        view_210 = torch.ops.aten.view.default(mul_523, [256, 128, 3, 3]);  mul_523 = None
        convolution_95 = torch.ops.aten.convolution.default(mul_520, view_210, arg44_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_520 = view_210 = arg44_1 = None
        mul_524 = torch.ops.aten.mul.Tensor(convolution_95, 0.5)
        mul_525 = torch.ops.aten.mul.Tensor(convolution_95, 0.7071067811865476);  convolution_95 = None
        erf_62 = torch.ops.aten.erf.default(mul_525);  mul_525 = None
        add_145 = torch.ops.aten.add.Tensor(erf_62, 1);  erf_62 = None
        mul_526 = torch.ops.aten.mul.Tensor(mul_524, add_145);  mul_524 = add_145 = None
        mul_527 = torch.ops.aten.mul.Tensor(mul_526, 1.7015043497085571);  mul_526 = None
        view_211 = torch.ops.aten.view.default(arg45_1, [1, 512, -1]);  arg45_1 = None
        mul_528 = torch.ops.aten.mul.Tensor(arg46_1, 0.0625);  arg46_1 = None
        view_212 = torch.ops.aten.view.default(mul_528, [-1]);  mul_528 = None
        var_mean_70 = torch.ops.aten.var_mean.correction(view_211, [0, 2], correction = 0, keepdim = True)
        getitem_140 = var_mean_70[0]
        getitem_141 = var_mean_70[1];  var_mean_70 = None
        add_146 = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
        rsqrt_70 = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_70 = torch.ops.aten.sub.Tensor(view_211, getitem_141);  view_211 = getitem_141 = None
        mul_529 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(view_212, -1);  view_212 = None
        mul_530 = torch.ops.aten.mul.Tensor(mul_529, unsqueeze_70);  mul_529 = unsqueeze_70 = None
        view_213 = torch.ops.aten.view.default(mul_530, [512, 256, 1, 1]);  mul_530 = None
        convolution_96 = torch.ops.aten.convolution.default(mul_527, view_213, arg47_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_527 = view_213 = arg47_1 = None
        mean_14 = torch.ops.aten.mean.dim(convolution_96, [2, 3], True)
        convolution_97 = torch.ops.aten.convolution.default(mean_14, arg48_1, arg49_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_14 = arg48_1 = arg49_1 = None
        relu_13 = torch.ops.aten.relu.default(convolution_97);  convolution_97 = None
        convolution_98 = torch.ops.aten.convolution.default(relu_13, arg50_1, arg51_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_13 = arg50_1 = arg51_1 = None
        sigmoid_13 = torch.ops.aten.sigmoid.default(convolution_98);  convolution_98 = None
        mul_531 = torch.ops.aten.mul.Tensor(convolution_96, sigmoid_13);  convolution_96 = sigmoid_13 = None
        mul_532 = torch.ops.aten.mul.Tensor(mul_531, 2.0);  mul_531 = None
        mul_533 = torch.ops.aten.mul.Tensor(mul_532, arg52_1);  mul_532 = arg52_1 = None
        mul_534 = torch.ops.aten.mul.Tensor(mul_533, 0.2);  mul_533 = None
        add_147 = torch.ops.aten.add.Tensor(mul_534, convolution_92);  mul_534 = convolution_92 = None
        mul_535 = torch.ops.aten.mul.Tensor(add_147, 0.5)
        mul_536 = torch.ops.aten.mul.Tensor(add_147, 0.7071067811865476)
        erf_63 = torch.ops.aten.erf.default(mul_536);  mul_536 = None
        add_148 = torch.ops.aten.add.Tensor(erf_63, 1);  erf_63 = None
        mul_537 = torch.ops.aten.mul.Tensor(mul_535, add_148);  mul_535 = add_148 = None
        mul_538 = torch.ops.aten.mul.Tensor(mul_537, 1.7015043497085571);  mul_537 = None
        mul_539 = torch.ops.aten.mul.Tensor(mul_538, 0.9805806756909201);  mul_538 = None
        view_214 = torch.ops.aten.view.default(arg53_1, [1, 256, -1]);  arg53_1 = None
        mul_540 = torch.ops.aten.mul.Tensor(arg54_1, 0.04419417382415922);  arg54_1 = None
        view_215 = torch.ops.aten.view.default(mul_540, [-1]);  mul_540 = None
        var_mean_71 = torch.ops.aten.var_mean.correction(view_214, [0, 2], correction = 0, keepdim = True)
        getitem_142 = var_mean_71[0]
        getitem_143 = var_mean_71[1];  var_mean_71 = None
        add_149 = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
        rsqrt_71 = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_71 = torch.ops.aten.sub.Tensor(view_214, getitem_143);  view_214 = getitem_143 = None
        mul_541 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(view_215, -1);  view_215 = None
        mul_542 = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_71);  mul_541 = unsqueeze_71 = None
        view_216 = torch.ops.aten.view.default(mul_542, [256, 512, 1, 1]);  mul_542 = None
        convolution_99 = torch.ops.aten.convolution.default(mul_539, view_216, arg55_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_539 = view_216 = arg55_1 = None
        mul_543 = torch.ops.aten.mul.Tensor(convolution_99, 0.5)
        mul_544 = torch.ops.aten.mul.Tensor(convolution_99, 0.7071067811865476);  convolution_99 = None
        erf_64 = torch.ops.aten.erf.default(mul_544);  mul_544 = None
        add_150 = torch.ops.aten.add.Tensor(erf_64, 1);  erf_64 = None
        mul_545 = torch.ops.aten.mul.Tensor(mul_543, add_150);  mul_543 = add_150 = None
        mul_546 = torch.ops.aten.mul.Tensor(mul_545, 1.7015043497085571);  mul_545 = None
        view_217 = torch.ops.aten.view.default(arg56_1, [1, 256, -1]);  arg56_1 = None
        mul_547 = torch.ops.aten.mul.Tensor(arg57_1, 0.02946278254943948);  arg57_1 = None
        view_218 = torch.ops.aten.view.default(mul_547, [-1]);  mul_547 = None
        var_mean_72 = torch.ops.aten.var_mean.correction(view_217, [0, 2], correction = 0, keepdim = True)
        getitem_144 = var_mean_72[0]
        getitem_145 = var_mean_72[1];  var_mean_72 = None
        add_151 = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
        rsqrt_72 = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_72 = torch.ops.aten.sub.Tensor(view_217, getitem_145);  view_217 = getitem_145 = None
        mul_548 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(view_218, -1);  view_218 = None
        mul_549 = torch.ops.aten.mul.Tensor(mul_548, unsqueeze_72);  mul_548 = unsqueeze_72 = None
        view_219 = torch.ops.aten.view.default(mul_549, [256, 128, 3, 3]);  mul_549 = None
        convolution_100 = torch.ops.aten.convolution.default(mul_546, view_219, arg58_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_546 = view_219 = arg58_1 = None
        mul_550 = torch.ops.aten.mul.Tensor(convolution_100, 0.5)
        mul_551 = torch.ops.aten.mul.Tensor(convolution_100, 0.7071067811865476);  convolution_100 = None
        erf_65 = torch.ops.aten.erf.default(mul_551);  mul_551 = None
        add_152 = torch.ops.aten.add.Tensor(erf_65, 1);  erf_65 = None
        mul_552 = torch.ops.aten.mul.Tensor(mul_550, add_152);  mul_550 = add_152 = None
        mul_553 = torch.ops.aten.mul.Tensor(mul_552, 1.7015043497085571);  mul_552 = None
        view_220 = torch.ops.aten.view.default(arg59_1, [1, 256, -1]);  arg59_1 = None
        mul_554 = torch.ops.aten.mul.Tensor(arg60_1, 0.02946278254943948);  arg60_1 = None
        view_221 = torch.ops.aten.view.default(mul_554, [-1]);  mul_554 = None
        var_mean_73 = torch.ops.aten.var_mean.correction(view_220, [0, 2], correction = 0, keepdim = True)
        getitem_146 = var_mean_73[0]
        getitem_147 = var_mean_73[1];  var_mean_73 = None
        add_153 = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_73 = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
        sub_73 = torch.ops.aten.sub.Tensor(view_220, getitem_147);  view_220 = getitem_147 = None
        mul_555 = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(view_221, -1);  view_221 = None
        mul_556 = torch.ops.aten.mul.Tensor(mul_555, unsqueeze_73);  mul_555 = unsqueeze_73 = None
        view_222 = torch.ops.aten.view.default(mul_556, [256, 128, 3, 3]);  mul_556 = None
        convolution_101 = torch.ops.aten.convolution.default(mul_553, view_222, arg61_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 2);  mul_553 = view_222 = arg61_1 = None
        mul_557 = torch.ops.aten.mul.Tensor(convolution_101, 0.5)
        mul_558 = torch.ops.aten.mul.Tensor(convolution_101, 0.7071067811865476);  convolution_101 = None
        erf_66 = torch.ops.aten.erf.default(mul_558);  mul_558 = None
        add_154 = torch.ops.aten.add.Tensor(erf_66, 1);  erf_66 = None
        mul_559 = torch.ops.aten.mul.Tensor(mul_557, add_154);  mul_557 = add_154 = None
        mul_560 = torch.ops.aten.mul.Tensor(mul_559, 1.7015043497085571);  mul_559 = None
        view_223 = torch.ops.aten.view.default(arg62_1, [1, 512, -1]);  arg62_1 = None
        mul_561 = torch.ops.aten.mul.Tensor(arg63_1, 0.0625);  arg63_1 = None
        view_224 = torch.ops.aten.view.default(mul_561, [-1]);  mul_561 = None
        var_mean_74 = torch.ops.aten.var_mean.correction(view_223, [0, 2], correction = 0, keepdim = True)
        getitem_148 = var_mean_74[0]
        getitem_149 = var_mean_74[1];  var_mean_74 = None
        add_155 = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
        rsqrt_74 = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_74 = torch.ops.aten.sub.Tensor(view_223, getitem_149);  view_223 = getitem_149 = None
        mul_562 = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(view_224, -1);  view_224 = None
        mul_563 = torch.ops.aten.mul.Tensor(mul_562, unsqueeze_74);  mul_562 = unsqueeze_74 = None
        view_225 = torch.ops.aten.view.default(mul_563, [512, 256, 1, 1]);  mul_563 = None
        convolution_102 = torch.ops.aten.convolution.default(mul_560, view_225, arg64_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_560 = view_225 = arg64_1 = None
        mean_15 = torch.ops.aten.mean.dim(convolution_102, [2, 3], True)
        convolution_103 = torch.ops.aten.convolution.default(mean_15, arg65_1, arg66_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_15 = arg65_1 = arg66_1 = None
        relu_14 = torch.ops.aten.relu.default(convolution_103);  convolution_103 = None
        convolution_104 = torch.ops.aten.convolution.default(relu_14, arg67_1, arg68_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg67_1 = arg68_1 = None
        sigmoid_14 = torch.ops.aten.sigmoid.default(convolution_104);  convolution_104 = None
        mul_564 = torch.ops.aten.mul.Tensor(convolution_102, sigmoid_14);  convolution_102 = sigmoid_14 = None
        mul_565 = torch.ops.aten.mul.Tensor(mul_564, 2.0);  mul_564 = None
        mul_566 = torch.ops.aten.mul.Tensor(mul_565, arg69_1);  mul_565 = arg69_1 = None
        mul_567 = torch.ops.aten.mul.Tensor(mul_566, 0.2);  mul_566 = None
        add_156 = torch.ops.aten.add.Tensor(mul_567, add_147);  mul_567 = add_147 = None
        mul_568 = torch.ops.aten.mul.Tensor(add_156, 0.5)
        mul_569 = torch.ops.aten.mul.Tensor(add_156, 0.7071067811865476);  add_156 = None
        erf_67 = torch.ops.aten.erf.default(mul_569);  mul_569 = None
        add_157 = torch.ops.aten.add.Tensor(erf_67, 1);  erf_67 = None
        mul_570 = torch.ops.aten.mul.Tensor(mul_568, add_157);  mul_568 = add_157 = None
        mul_571 = torch.ops.aten.mul.Tensor(mul_570, 1.7015043497085571);  mul_570 = None
        mul_572 = torch.ops.aten.mul.Tensor(mul_571, 0.9622504486493761);  mul_571 = None
        avg_pool2d_4 = torch.ops.aten.avg_pool2d.default(mul_572, [2, 2], [2, 2], [0, 0], True, False)
        view_226 = torch.ops.aten.view.default(arg70_1, [1, 1536, -1]);  arg70_1 = None
        mul_573 = torch.ops.aten.mul.Tensor(arg71_1, 0.04419417382415922);  arg71_1 = None
        view_227 = torch.ops.aten.view.default(mul_573, [-1]);  mul_573 = None
        var_mean_75 = torch.ops.aten.var_mean.correction(view_226, [0, 2], correction = 0, keepdim = True)
        getitem_150 = var_mean_75[0]
        getitem_151 = var_mean_75[1];  var_mean_75 = None
        add_158 = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
        rsqrt_75 = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_75 = torch.ops.aten.sub.Tensor(view_226, getitem_151);  view_226 = getitem_151 = None
        mul_574 = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(view_227, -1);  view_227 = None
        mul_575 = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_75);  mul_574 = unsqueeze_75 = None
        view_228 = torch.ops.aten.view.default(mul_575, [1536, 512, 1, 1]);  mul_575 = None
        convolution_105 = torch.ops.aten.convolution.default(avg_pool2d_4, view_228, arg72_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_4 = view_228 = arg72_1 = None
        view_229 = torch.ops.aten.view.default(arg73_1, [1, 768, -1]);  arg73_1 = None
        mul_576 = torch.ops.aten.mul.Tensor(arg74_1, 0.04419417382415922);  arg74_1 = None
        view_230 = torch.ops.aten.view.default(mul_576, [-1]);  mul_576 = None
        var_mean_76 = torch.ops.aten.var_mean.correction(view_229, [0, 2], correction = 0, keepdim = True)
        getitem_152 = var_mean_76[0]
        getitem_153 = var_mean_76[1];  var_mean_76 = None
        add_159 = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
        rsqrt_76 = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_76 = torch.ops.aten.sub.Tensor(view_229, getitem_153);  view_229 = getitem_153 = None
        mul_577 = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(view_230, -1);  view_230 = None
        mul_578 = torch.ops.aten.mul.Tensor(mul_577, unsqueeze_76);  mul_577 = unsqueeze_76 = None
        view_231 = torch.ops.aten.view.default(mul_578, [768, 512, 1, 1]);  mul_578 = None
        convolution_106 = torch.ops.aten.convolution.default(mul_572, view_231, arg75_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_572 = view_231 = arg75_1 = None
        mul_579 = torch.ops.aten.mul.Tensor(convolution_106, 0.5)
        mul_580 = torch.ops.aten.mul.Tensor(convolution_106, 0.7071067811865476);  convolution_106 = None
        erf_68 = torch.ops.aten.erf.default(mul_580);  mul_580 = None
        add_160 = torch.ops.aten.add.Tensor(erf_68, 1);  erf_68 = None
        mul_581 = torch.ops.aten.mul.Tensor(mul_579, add_160);  mul_579 = add_160 = None
        mul_582 = torch.ops.aten.mul.Tensor(mul_581, 1.7015043497085571);  mul_581 = None
        constant_pad_nd_8 = torch.ops.aten.constant_pad_nd.default(mul_582, [0, 1, 0, 1], 0.0);  mul_582 = None
        view_232 = torch.ops.aten.view.default(arg76_1, [1, 768, -1]);  arg76_1 = None
        mul_583 = torch.ops.aten.mul.Tensor(arg77_1, 0.02946278254943948);  arg77_1 = None
        view_233 = torch.ops.aten.view.default(mul_583, [-1]);  mul_583 = None
        var_mean_77 = torch.ops.aten.var_mean.correction(view_232, [0, 2], correction = 0, keepdim = True)
        getitem_154 = var_mean_77[0]
        getitem_155 = var_mean_77[1];  var_mean_77 = None
        add_161 = torch.ops.aten.add.Tensor(getitem_154, 1e-05);  getitem_154 = None
        rsqrt_77 = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_77 = torch.ops.aten.sub.Tensor(view_232, getitem_155);  view_232 = getitem_155 = None
        mul_584 = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(view_233, -1);  view_233 = None
        mul_585 = torch.ops.aten.mul.Tensor(mul_584, unsqueeze_77);  mul_584 = unsqueeze_77 = None
        view_234 = torch.ops.aten.view.default(mul_585, [768, 128, 3, 3]);  mul_585 = None
        convolution_107 = torch.ops.aten.convolution.default(constant_pad_nd_8, view_234, arg78_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6);  constant_pad_nd_8 = view_234 = arg78_1 = None
        mul_586 = torch.ops.aten.mul.Tensor(convolution_107, 0.5)
        mul_587 = torch.ops.aten.mul.Tensor(convolution_107, 0.7071067811865476);  convolution_107 = None
        erf_69 = torch.ops.aten.erf.default(mul_587);  mul_587 = None
        add_162 = torch.ops.aten.add.Tensor(erf_69, 1);  erf_69 = None
        mul_588 = torch.ops.aten.mul.Tensor(mul_586, add_162);  mul_586 = add_162 = None
        mul_589 = torch.ops.aten.mul.Tensor(mul_588, 1.7015043497085571);  mul_588 = None
        view_235 = torch.ops.aten.view.default(arg79_1, [1, 768, -1]);  arg79_1 = None
        mul_590 = torch.ops.aten.mul.Tensor(arg80_1, 0.02946278254943948);  arg80_1 = None
        view_236 = torch.ops.aten.view.default(mul_590, [-1]);  mul_590 = None
        var_mean_78 = torch.ops.aten.var_mean.correction(view_235, [0, 2], correction = 0, keepdim = True)
        getitem_156 = var_mean_78[0]
        getitem_157 = var_mean_78[1];  var_mean_78 = None
        add_163 = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_78 = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        sub_78 = torch.ops.aten.sub.Tensor(view_235, getitem_157);  view_235 = getitem_157 = None
        mul_591 = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(view_236, -1);  view_236 = None
        mul_592 = torch.ops.aten.mul.Tensor(mul_591, unsqueeze_78);  mul_591 = unsqueeze_78 = None
        view_237 = torch.ops.aten.view.default(mul_592, [768, 128, 3, 3]);  mul_592 = None
        convolution_108 = torch.ops.aten.convolution.default(mul_589, view_237, arg81_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_589 = view_237 = arg81_1 = None
        mul_593 = torch.ops.aten.mul.Tensor(convolution_108, 0.5)
        mul_594 = torch.ops.aten.mul.Tensor(convolution_108, 0.7071067811865476);  convolution_108 = None
        erf_70 = torch.ops.aten.erf.default(mul_594);  mul_594 = None
        add_164 = torch.ops.aten.add.Tensor(erf_70, 1);  erf_70 = None
        mul_595 = torch.ops.aten.mul.Tensor(mul_593, add_164);  mul_593 = add_164 = None
        mul_596 = torch.ops.aten.mul.Tensor(mul_595, 1.7015043497085571);  mul_595 = None
        view_238 = torch.ops.aten.view.default(arg82_1, [1, 1536, -1]);  arg82_1 = None
        mul_597 = torch.ops.aten.mul.Tensor(arg83_1, 0.03608439182435161);  arg83_1 = None
        view_239 = torch.ops.aten.view.default(mul_597, [-1]);  mul_597 = None
        var_mean_79 = torch.ops.aten.var_mean.correction(view_238, [0, 2], correction = 0, keepdim = True)
        getitem_158 = var_mean_79[0]
        getitem_159 = var_mean_79[1];  var_mean_79 = None
        add_165 = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_79 = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_79 = torch.ops.aten.sub.Tensor(view_238, getitem_159);  view_238 = getitem_159 = None
        mul_598 = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(view_239, -1);  view_239 = None
        mul_599 = torch.ops.aten.mul.Tensor(mul_598, unsqueeze_79);  mul_598 = unsqueeze_79 = None
        view_240 = torch.ops.aten.view.default(mul_599, [1536, 768, 1, 1]);  mul_599 = None
        convolution_109 = torch.ops.aten.convolution.default(mul_596, view_240, arg84_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_596 = view_240 = arg84_1 = None
        mean_16 = torch.ops.aten.mean.dim(convolution_109, [2, 3], True)
        convolution_110 = torch.ops.aten.convolution.default(mean_16, arg85_1, arg86_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_16 = arg85_1 = arg86_1 = None
        relu_15 = torch.ops.aten.relu.default(convolution_110);  convolution_110 = None
        convolution_111 = torch.ops.aten.convolution.default(relu_15, arg87_1, arg88_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_15 = arg87_1 = arg88_1 = None
        sigmoid_15 = torch.ops.aten.sigmoid.default(convolution_111);  convolution_111 = None
        mul_600 = torch.ops.aten.mul.Tensor(convolution_109, sigmoid_15);  convolution_109 = sigmoid_15 = None
        mul_601 = torch.ops.aten.mul.Tensor(mul_600, 2.0);  mul_600 = None
        mul_602 = torch.ops.aten.mul.Tensor(mul_601, arg89_1);  mul_601 = arg89_1 = None
        mul_603 = torch.ops.aten.mul.Tensor(mul_602, 0.2);  mul_602 = None
        add_166 = torch.ops.aten.add.Tensor(mul_603, convolution_105);  mul_603 = convolution_105 = None
        mul_604 = torch.ops.aten.mul.Tensor(add_166, 0.5)
        mul_605 = torch.ops.aten.mul.Tensor(add_166, 0.7071067811865476)
        erf_71 = torch.ops.aten.erf.default(mul_605);  mul_605 = None
        add_167 = torch.ops.aten.add.Tensor(erf_71, 1);  erf_71 = None
        mul_606 = torch.ops.aten.mul.Tensor(mul_604, add_167);  mul_604 = add_167 = None
        mul_607 = torch.ops.aten.mul.Tensor(mul_606, 1.7015043497085571);  mul_606 = None
        mul_608 = torch.ops.aten.mul.Tensor(mul_607, 0.9805806756909201);  mul_607 = None
        view_241 = torch.ops.aten.view.default(arg90_1, [1, 768, -1]);  arg90_1 = None
        mul_609 = torch.ops.aten.mul.Tensor(arg91_1, 0.02551551815399144);  arg91_1 = None
        view_242 = torch.ops.aten.view.default(mul_609, [-1]);  mul_609 = None
        var_mean_80 = torch.ops.aten.var_mean.correction(view_241, [0, 2], correction = 0, keepdim = True)
        getitem_160 = var_mean_80[0]
        getitem_161 = var_mean_80[1];  var_mean_80 = None
        add_168 = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_80 = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        sub_80 = torch.ops.aten.sub.Tensor(view_241, getitem_161);  view_241 = getitem_161 = None
        mul_610 = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(view_242, -1);  view_242 = None
        mul_611 = torch.ops.aten.mul.Tensor(mul_610, unsqueeze_80);  mul_610 = unsqueeze_80 = None
        view_243 = torch.ops.aten.view.default(mul_611, [768, 1536, 1, 1]);  mul_611 = None
        convolution_112 = torch.ops.aten.convolution.default(mul_608, view_243, arg92_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_608 = view_243 = arg92_1 = None
        mul_612 = torch.ops.aten.mul.Tensor(convolution_112, 0.5)
        mul_613 = torch.ops.aten.mul.Tensor(convolution_112, 0.7071067811865476);  convolution_112 = None
        erf_72 = torch.ops.aten.erf.default(mul_613);  mul_613 = None
        add_169 = torch.ops.aten.add.Tensor(erf_72, 1);  erf_72 = None
        mul_614 = torch.ops.aten.mul.Tensor(mul_612, add_169);  mul_612 = add_169 = None
        mul_615 = torch.ops.aten.mul.Tensor(mul_614, 1.7015043497085571);  mul_614 = None
        view_244 = torch.ops.aten.view.default(arg93_1, [1, 768, -1]);  arg93_1 = None
        mul_616 = torch.ops.aten.mul.Tensor(arg94_1, 0.02946278254943948);  arg94_1 = None
        view_245 = torch.ops.aten.view.default(mul_616, [-1]);  mul_616 = None
        var_mean_81 = torch.ops.aten.var_mean.correction(view_244, [0, 2], correction = 0, keepdim = True)
        getitem_162 = var_mean_81[0]
        getitem_163 = var_mean_81[1];  var_mean_81 = None
        add_170 = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_81 = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_81 = torch.ops.aten.sub.Tensor(view_244, getitem_163);  view_244 = getitem_163 = None
        mul_617 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = rsqrt_81 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(view_245, -1);  view_245 = None
        mul_618 = torch.ops.aten.mul.Tensor(mul_617, unsqueeze_81);  mul_617 = unsqueeze_81 = None
        view_246 = torch.ops.aten.view.default(mul_618, [768, 128, 3, 3]);  mul_618 = None
        convolution_113 = torch.ops.aten.convolution.default(mul_615, view_246, arg95_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_615 = view_246 = arg95_1 = None
        mul_619 = torch.ops.aten.mul.Tensor(convolution_113, 0.5)
        mul_620 = torch.ops.aten.mul.Tensor(convolution_113, 0.7071067811865476);  convolution_113 = None
        erf_73 = torch.ops.aten.erf.default(mul_620);  mul_620 = None
        add_171 = torch.ops.aten.add.Tensor(erf_73, 1);  erf_73 = None
        mul_621 = torch.ops.aten.mul.Tensor(mul_619, add_171);  mul_619 = add_171 = None
        mul_622 = torch.ops.aten.mul.Tensor(mul_621, 1.7015043497085571);  mul_621 = None
        view_247 = torch.ops.aten.view.default(arg96_1, [1, 768, -1]);  arg96_1 = None
        mul_623 = torch.ops.aten.mul.Tensor(arg97_1, 0.02946278254943948);  arg97_1 = None
        view_248 = torch.ops.aten.view.default(mul_623, [-1]);  mul_623 = None
        var_mean_82 = torch.ops.aten.var_mean.correction(view_247, [0, 2], correction = 0, keepdim = True)
        getitem_164 = var_mean_82[0]
        getitem_165 = var_mean_82[1];  var_mean_82 = None
        add_172 = torch.ops.aten.add.Tensor(getitem_164, 1e-05);  getitem_164 = None
        rsqrt_82 = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_82 = torch.ops.aten.sub.Tensor(view_247, getitem_165);  view_247 = getitem_165 = None
        mul_624 = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = rsqrt_82 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(view_248, -1);  view_248 = None
        mul_625 = torch.ops.aten.mul.Tensor(mul_624, unsqueeze_82);  mul_624 = unsqueeze_82 = None
        view_249 = torch.ops.aten.view.default(mul_625, [768, 128, 3, 3]);  mul_625 = None
        convolution_114 = torch.ops.aten.convolution.default(mul_622, view_249, arg98_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_622 = view_249 = arg98_1 = None
        mul_626 = torch.ops.aten.mul.Tensor(convolution_114, 0.5)
        mul_627 = torch.ops.aten.mul.Tensor(convolution_114, 0.7071067811865476);  convolution_114 = None
        erf_74 = torch.ops.aten.erf.default(mul_627);  mul_627 = None
        add_173 = torch.ops.aten.add.Tensor(erf_74, 1);  erf_74 = None
        mul_628 = torch.ops.aten.mul.Tensor(mul_626, add_173);  mul_626 = add_173 = None
        mul_629 = torch.ops.aten.mul.Tensor(mul_628, 1.7015043497085571);  mul_628 = None
        view_250 = torch.ops.aten.view.default(arg99_1, [1, 1536, -1]);  arg99_1 = None
        mul_630 = torch.ops.aten.mul.Tensor(arg100_1, 0.03608439182435161);  arg100_1 = None
        view_251 = torch.ops.aten.view.default(mul_630, [-1]);  mul_630 = None
        var_mean_83 = torch.ops.aten.var_mean.correction(view_250, [0, 2], correction = 0, keepdim = True)
        getitem_166 = var_mean_83[0]
        getitem_167 = var_mean_83[1];  var_mean_83 = None
        add_174 = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
        rsqrt_83 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_83 = torch.ops.aten.sub.Tensor(view_250, getitem_167);  view_250 = getitem_167 = None
        mul_631 = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = rsqrt_83 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(view_251, -1);  view_251 = None
        mul_632 = torch.ops.aten.mul.Tensor(mul_631, unsqueeze_83);  mul_631 = unsqueeze_83 = None
        view_252 = torch.ops.aten.view.default(mul_632, [1536, 768, 1, 1]);  mul_632 = None
        convolution_115 = torch.ops.aten.convolution.default(mul_629, view_252, arg101_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_629 = view_252 = arg101_1 = None
        mean_17 = torch.ops.aten.mean.dim(convolution_115, [2, 3], True)
        convolution_116 = torch.ops.aten.convolution.default(mean_17, arg102_1, arg103_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_17 = arg102_1 = arg103_1 = None
        relu_16 = torch.ops.aten.relu.default(convolution_116);  convolution_116 = None
        convolution_117 = torch.ops.aten.convolution.default(relu_16, arg104_1, arg105_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_16 = arg104_1 = arg105_1 = None
        sigmoid_16 = torch.ops.aten.sigmoid.default(convolution_117);  convolution_117 = None
        mul_633 = torch.ops.aten.mul.Tensor(convolution_115, sigmoid_16);  convolution_115 = sigmoid_16 = None
        mul_634 = torch.ops.aten.mul.Tensor(mul_633, 2.0);  mul_633 = None
        mul_635 = torch.ops.aten.mul.Tensor(mul_634, arg106_1);  mul_634 = arg106_1 = None
        mul_636 = torch.ops.aten.mul.Tensor(mul_635, 0.2);  mul_635 = None
        add_175 = torch.ops.aten.add.Tensor(mul_636, add_166);  mul_636 = add_166 = None
        mul_637 = torch.ops.aten.mul.Tensor(add_175, 0.5)
        mul_638 = torch.ops.aten.mul.Tensor(add_175, 0.7071067811865476)
        erf_75 = torch.ops.aten.erf.default(mul_638);  mul_638 = None
        add_176 = torch.ops.aten.add.Tensor(erf_75, 1);  erf_75 = None
        mul_639 = torch.ops.aten.mul.Tensor(mul_637, add_176);  mul_637 = add_176 = None
        mul_640 = torch.ops.aten.mul.Tensor(mul_639, 1.7015043497085571);  mul_639 = None
        mul_641 = torch.ops.aten.mul.Tensor(mul_640, 0.9622504486493761);  mul_640 = None
        view_253 = torch.ops.aten.view.default(arg107_1, [1, 768, -1]);  arg107_1 = None
        mul_642 = torch.ops.aten.mul.Tensor(arg108_1, 0.02551551815399144);  arg108_1 = None
        view_254 = torch.ops.aten.view.default(mul_642, [-1]);  mul_642 = None
        var_mean_84 = torch.ops.aten.var_mean.correction(view_253, [0, 2], correction = 0, keepdim = True)
        getitem_168 = var_mean_84[0]
        getitem_169 = var_mean_84[1];  var_mean_84 = None
        add_177 = torch.ops.aten.add.Tensor(getitem_168, 1e-05);  getitem_168 = None
        rsqrt_84 = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        sub_84 = torch.ops.aten.sub.Tensor(view_253, getitem_169);  view_253 = getitem_169 = None
        mul_643 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = rsqrt_84 = None
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(view_254, -1);  view_254 = None
        mul_644 = torch.ops.aten.mul.Tensor(mul_643, unsqueeze_84);  mul_643 = unsqueeze_84 = None
        view_255 = torch.ops.aten.view.default(mul_644, [768, 1536, 1, 1]);  mul_644 = None
        convolution_118 = torch.ops.aten.convolution.default(mul_641, view_255, arg109_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_641 = view_255 = arg109_1 = None
        mul_645 = torch.ops.aten.mul.Tensor(convolution_118, 0.5)
        mul_646 = torch.ops.aten.mul.Tensor(convolution_118, 0.7071067811865476);  convolution_118 = None
        erf_76 = torch.ops.aten.erf.default(mul_646);  mul_646 = None
        add_178 = torch.ops.aten.add.Tensor(erf_76, 1);  erf_76 = None
        mul_647 = torch.ops.aten.mul.Tensor(mul_645, add_178);  mul_645 = add_178 = None
        mul_648 = torch.ops.aten.mul.Tensor(mul_647, 1.7015043497085571);  mul_647 = None
        view_256 = torch.ops.aten.view.default(arg110_1, [1, 768, -1]);  arg110_1 = None
        mul_649 = torch.ops.aten.mul.Tensor(arg111_1, 0.02946278254943948);  arg111_1 = None
        view_257 = torch.ops.aten.view.default(mul_649, [-1]);  mul_649 = None
        var_mean_85 = torch.ops.aten.var_mean.correction(view_256, [0, 2], correction = 0, keepdim = True)
        getitem_170 = var_mean_85[0]
        getitem_171 = var_mean_85[1];  var_mean_85 = None
        add_179 = torch.ops.aten.add.Tensor(getitem_170, 1e-05);  getitem_170 = None
        rsqrt_85 = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        sub_85 = torch.ops.aten.sub.Tensor(view_256, getitem_171);  view_256 = getitem_171 = None
        mul_650 = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = rsqrt_85 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(view_257, -1);  view_257 = None
        mul_651 = torch.ops.aten.mul.Tensor(mul_650, unsqueeze_85);  mul_650 = unsqueeze_85 = None
        view_258 = torch.ops.aten.view.default(mul_651, [768, 128, 3, 3]);  mul_651 = None
        convolution_119 = torch.ops.aten.convolution.default(mul_648, view_258, arg112_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_648 = view_258 = arg112_1 = None
        mul_652 = torch.ops.aten.mul.Tensor(convolution_119, 0.5)
        mul_653 = torch.ops.aten.mul.Tensor(convolution_119, 0.7071067811865476);  convolution_119 = None
        erf_77 = torch.ops.aten.erf.default(mul_653);  mul_653 = None
        add_180 = torch.ops.aten.add.Tensor(erf_77, 1);  erf_77 = None
        mul_654 = torch.ops.aten.mul.Tensor(mul_652, add_180);  mul_652 = add_180 = None
        mul_655 = torch.ops.aten.mul.Tensor(mul_654, 1.7015043497085571);  mul_654 = None
        view_259 = torch.ops.aten.view.default(arg113_1, [1, 768, -1]);  arg113_1 = None
        mul_656 = torch.ops.aten.mul.Tensor(arg114_1, 0.02946278254943948);  arg114_1 = None
        view_260 = torch.ops.aten.view.default(mul_656, [-1]);  mul_656 = None
        var_mean_86 = torch.ops.aten.var_mean.correction(view_259, [0, 2], correction = 0, keepdim = True)
        getitem_172 = var_mean_86[0]
        getitem_173 = var_mean_86[1];  var_mean_86 = None
        add_181 = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
        rsqrt_86 = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_86 = torch.ops.aten.sub.Tensor(view_259, getitem_173);  view_259 = getitem_173 = None
        mul_657 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = rsqrt_86 = None
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(view_260, -1);  view_260 = None
        mul_658 = torch.ops.aten.mul.Tensor(mul_657, unsqueeze_86);  mul_657 = unsqueeze_86 = None
        view_261 = torch.ops.aten.view.default(mul_658, [768, 128, 3, 3]);  mul_658 = None
        convolution_120 = torch.ops.aten.convolution.default(mul_655, view_261, arg115_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_655 = view_261 = arg115_1 = None
        mul_659 = torch.ops.aten.mul.Tensor(convolution_120, 0.5)
        mul_660 = torch.ops.aten.mul.Tensor(convolution_120, 0.7071067811865476);  convolution_120 = None
        erf_78 = torch.ops.aten.erf.default(mul_660);  mul_660 = None
        add_182 = torch.ops.aten.add.Tensor(erf_78, 1);  erf_78 = None
        mul_661 = torch.ops.aten.mul.Tensor(mul_659, add_182);  mul_659 = add_182 = None
        mul_662 = torch.ops.aten.mul.Tensor(mul_661, 1.7015043497085571);  mul_661 = None
        view_262 = torch.ops.aten.view.default(arg116_1, [1, 1536, -1]);  arg116_1 = None
        mul_663 = torch.ops.aten.mul.Tensor(arg117_1, 0.03608439182435161);  arg117_1 = None
        view_263 = torch.ops.aten.view.default(mul_663, [-1]);  mul_663 = None
        var_mean_87 = torch.ops.aten.var_mean.correction(view_262, [0, 2], correction = 0, keepdim = True)
        getitem_174 = var_mean_87[0]
        getitem_175 = var_mean_87[1];  var_mean_87 = None
        add_183 = torch.ops.aten.add.Tensor(getitem_174, 1e-05);  getitem_174 = None
        rsqrt_87 = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        sub_87 = torch.ops.aten.sub.Tensor(view_262, getitem_175);  view_262 = getitem_175 = None
        mul_664 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = rsqrt_87 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(view_263, -1);  view_263 = None
        mul_665 = torch.ops.aten.mul.Tensor(mul_664, unsqueeze_87);  mul_664 = unsqueeze_87 = None
        view_264 = torch.ops.aten.view.default(mul_665, [1536, 768, 1, 1]);  mul_665 = None
        convolution_121 = torch.ops.aten.convolution.default(mul_662, view_264, arg118_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_662 = view_264 = arg118_1 = None
        mean_18 = torch.ops.aten.mean.dim(convolution_121, [2, 3], True)
        convolution_122 = torch.ops.aten.convolution.default(mean_18, arg119_1, arg120_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_18 = arg119_1 = arg120_1 = None
        relu_17 = torch.ops.aten.relu.default(convolution_122);  convolution_122 = None
        convolution_123 = torch.ops.aten.convolution.default(relu_17, arg121_1, arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg121_1 = arg122_1 = None
        sigmoid_17 = torch.ops.aten.sigmoid.default(convolution_123);  convolution_123 = None
        mul_666 = torch.ops.aten.mul.Tensor(convolution_121, sigmoid_17);  convolution_121 = sigmoid_17 = None
        mul_667 = torch.ops.aten.mul.Tensor(mul_666, 2.0);  mul_666 = None
        mul_668 = torch.ops.aten.mul.Tensor(mul_667, arg123_1);  mul_667 = arg123_1 = None
        mul_669 = torch.ops.aten.mul.Tensor(mul_668, 0.2);  mul_668 = None
        add_184 = torch.ops.aten.add.Tensor(mul_669, add_175);  mul_669 = add_175 = None
        mul_670 = torch.ops.aten.mul.Tensor(add_184, 0.5)
        mul_671 = torch.ops.aten.mul.Tensor(add_184, 0.7071067811865476)
        erf_79 = torch.ops.aten.erf.default(mul_671);  mul_671 = None
        add_185 = torch.ops.aten.add.Tensor(erf_79, 1);  erf_79 = None
        mul_672 = torch.ops.aten.mul.Tensor(mul_670, add_185);  mul_670 = add_185 = None
        mul_673 = torch.ops.aten.mul.Tensor(mul_672, 1.7015043497085571);  mul_672 = None
        mul_674 = torch.ops.aten.mul.Tensor(mul_673, 0.9449111825230679);  mul_673 = None
        view_265 = torch.ops.aten.view.default(arg124_1, [1, 768, -1]);  arg124_1 = None
        mul_675 = torch.ops.aten.mul.Tensor(arg125_1, 0.02551551815399144);  arg125_1 = None
        view_266 = torch.ops.aten.view.default(mul_675, [-1]);  mul_675 = None
        var_mean_88 = torch.ops.aten.var_mean.correction(view_265, [0, 2], correction = 0, keepdim = True)
        getitem_176 = var_mean_88[0]
        getitem_177 = var_mean_88[1];  var_mean_88 = None
        add_186 = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_88 = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_88 = torch.ops.aten.sub.Tensor(view_265, getitem_177);  view_265 = getitem_177 = None
        mul_676 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = rsqrt_88 = None
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(view_266, -1);  view_266 = None
        mul_677 = torch.ops.aten.mul.Tensor(mul_676, unsqueeze_88);  mul_676 = unsqueeze_88 = None
        view_267 = torch.ops.aten.view.default(mul_677, [768, 1536, 1, 1]);  mul_677 = None
        convolution_124 = torch.ops.aten.convolution.default(mul_674, view_267, arg126_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_674 = view_267 = arg126_1 = None
        mul_678 = torch.ops.aten.mul.Tensor(convolution_124, 0.5)
        mul_679 = torch.ops.aten.mul.Tensor(convolution_124, 0.7071067811865476);  convolution_124 = None
        erf_80 = torch.ops.aten.erf.default(mul_679);  mul_679 = None
        add_187 = torch.ops.aten.add.Tensor(erf_80, 1);  erf_80 = None
        mul_680 = torch.ops.aten.mul.Tensor(mul_678, add_187);  mul_678 = add_187 = None
        mul_681 = torch.ops.aten.mul.Tensor(mul_680, 1.7015043497085571);  mul_680 = None
        view_268 = torch.ops.aten.view.default(arg127_1, [1, 768, -1]);  arg127_1 = None
        mul_682 = torch.ops.aten.mul.Tensor(arg128_1, 0.02946278254943948);  arg128_1 = None
        view_269 = torch.ops.aten.view.default(mul_682, [-1]);  mul_682 = None
        var_mean_89 = torch.ops.aten.var_mean.correction(view_268, [0, 2], correction = 0, keepdim = True)
        getitem_178 = var_mean_89[0]
        getitem_179 = var_mean_89[1];  var_mean_89 = None
        add_188 = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
        rsqrt_89 = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_89 = torch.ops.aten.sub.Tensor(view_268, getitem_179);  view_268 = getitem_179 = None
        mul_683 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = rsqrt_89 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(view_269, -1);  view_269 = None
        mul_684 = torch.ops.aten.mul.Tensor(mul_683, unsqueeze_89);  mul_683 = unsqueeze_89 = None
        view_270 = torch.ops.aten.view.default(mul_684, [768, 128, 3, 3]);  mul_684 = None
        convolution_125 = torch.ops.aten.convolution.default(mul_681, view_270, arg129_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_681 = view_270 = arg129_1 = None
        mul_685 = torch.ops.aten.mul.Tensor(convolution_125, 0.5)
        mul_686 = torch.ops.aten.mul.Tensor(convolution_125, 0.7071067811865476);  convolution_125 = None
        erf_81 = torch.ops.aten.erf.default(mul_686);  mul_686 = None
        add_189 = torch.ops.aten.add.Tensor(erf_81, 1);  erf_81 = None
        mul_687 = torch.ops.aten.mul.Tensor(mul_685, add_189);  mul_685 = add_189 = None
        mul_688 = torch.ops.aten.mul.Tensor(mul_687, 1.7015043497085571);  mul_687 = None
        view_271 = torch.ops.aten.view.default(arg130_1, [1, 768, -1]);  arg130_1 = None
        mul_689 = torch.ops.aten.mul.Tensor(arg131_1, 0.02946278254943948);  arg131_1 = None
        view_272 = torch.ops.aten.view.default(mul_689, [-1]);  mul_689 = None
        var_mean_90 = torch.ops.aten.var_mean.correction(view_271, [0, 2], correction = 0, keepdim = True)
        getitem_180 = var_mean_90[0]
        getitem_181 = var_mean_90[1];  var_mean_90 = None
        add_190 = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_90 = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_90 = torch.ops.aten.sub.Tensor(view_271, getitem_181);  view_271 = getitem_181 = None
        mul_690 = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = rsqrt_90 = None
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(view_272, -1);  view_272 = None
        mul_691 = torch.ops.aten.mul.Tensor(mul_690, unsqueeze_90);  mul_690 = unsqueeze_90 = None
        view_273 = torch.ops.aten.view.default(mul_691, [768, 128, 3, 3]);  mul_691 = None
        convolution_126 = torch.ops.aten.convolution.default(mul_688, view_273, arg132_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_688 = view_273 = arg132_1 = None
        mul_692 = torch.ops.aten.mul.Tensor(convolution_126, 0.5)
        mul_693 = torch.ops.aten.mul.Tensor(convolution_126, 0.7071067811865476);  convolution_126 = None
        erf_82 = torch.ops.aten.erf.default(mul_693);  mul_693 = None
        add_191 = torch.ops.aten.add.Tensor(erf_82, 1);  erf_82 = None
        mul_694 = torch.ops.aten.mul.Tensor(mul_692, add_191);  mul_692 = add_191 = None
        mul_695 = torch.ops.aten.mul.Tensor(mul_694, 1.7015043497085571);  mul_694 = None
        view_274 = torch.ops.aten.view.default(arg133_1, [1, 1536, -1]);  arg133_1 = None
        mul_696 = torch.ops.aten.mul.Tensor(arg134_1, 0.03608439182435161);  arg134_1 = None
        view_275 = torch.ops.aten.view.default(mul_696, [-1]);  mul_696 = None
        var_mean_91 = torch.ops.aten.var_mean.correction(view_274, [0, 2], correction = 0, keepdim = True)
        getitem_182 = var_mean_91[0]
        getitem_183 = var_mean_91[1];  var_mean_91 = None
        add_192 = torch.ops.aten.add.Tensor(getitem_182, 1e-05);  getitem_182 = None
        rsqrt_91 = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_91 = torch.ops.aten.sub.Tensor(view_274, getitem_183);  view_274 = getitem_183 = None
        mul_697 = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = rsqrt_91 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(view_275, -1);  view_275 = None
        mul_698 = torch.ops.aten.mul.Tensor(mul_697, unsqueeze_91);  mul_697 = unsqueeze_91 = None
        view_276 = torch.ops.aten.view.default(mul_698, [1536, 768, 1, 1]);  mul_698 = None
        convolution_127 = torch.ops.aten.convolution.default(mul_695, view_276, arg135_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_695 = view_276 = arg135_1 = None
        mean_19 = torch.ops.aten.mean.dim(convolution_127, [2, 3], True)
        convolution_128 = torch.ops.aten.convolution.default(mean_19, arg136_1, arg137_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_19 = arg136_1 = arg137_1 = None
        relu_18 = torch.ops.aten.relu.default(convolution_128);  convolution_128 = None
        convolution_129 = torch.ops.aten.convolution.default(relu_18, arg138_1, arg139_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_18 = arg138_1 = arg139_1 = None
        sigmoid_18 = torch.ops.aten.sigmoid.default(convolution_129);  convolution_129 = None
        mul_699 = torch.ops.aten.mul.Tensor(convolution_127, sigmoid_18);  convolution_127 = sigmoid_18 = None
        mul_700 = torch.ops.aten.mul.Tensor(mul_699, 2.0);  mul_699 = None
        mul_701 = torch.ops.aten.mul.Tensor(mul_700, arg140_1);  mul_700 = arg140_1 = None
        mul_702 = torch.ops.aten.mul.Tensor(mul_701, 0.2);  mul_701 = None
        add_193 = torch.ops.aten.add.Tensor(mul_702, add_184);  mul_702 = add_184 = None
        mul_703 = torch.ops.aten.mul.Tensor(add_193, 0.5)
        mul_704 = torch.ops.aten.mul.Tensor(add_193, 0.7071067811865476)
        erf_83 = torch.ops.aten.erf.default(mul_704);  mul_704 = None
        add_194 = torch.ops.aten.add.Tensor(erf_83, 1);  erf_83 = None
        mul_705 = torch.ops.aten.mul.Tensor(mul_703, add_194);  mul_703 = add_194 = None
        mul_706 = torch.ops.aten.mul.Tensor(mul_705, 1.7015043497085571);  mul_705 = None
        mul_707 = torch.ops.aten.mul.Tensor(mul_706, 0.9284766908852592);  mul_706 = None
        view_277 = torch.ops.aten.view.default(arg141_1, [1, 768, -1]);  arg141_1 = None
        mul_708 = torch.ops.aten.mul.Tensor(arg142_1, 0.02551551815399144);  arg142_1 = None
        view_278 = torch.ops.aten.view.default(mul_708, [-1]);  mul_708 = None
        var_mean_92 = torch.ops.aten.var_mean.correction(view_277, [0, 2], correction = 0, keepdim = True)
        getitem_184 = var_mean_92[0]
        getitem_185 = var_mean_92[1];  var_mean_92 = None
        add_195 = torch.ops.aten.add.Tensor(getitem_184, 1e-05);  getitem_184 = None
        rsqrt_92 = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        sub_92 = torch.ops.aten.sub.Tensor(view_277, getitem_185);  view_277 = getitem_185 = None
        mul_709 = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = rsqrt_92 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(view_278, -1);  view_278 = None
        mul_710 = torch.ops.aten.mul.Tensor(mul_709, unsqueeze_92);  mul_709 = unsqueeze_92 = None
        view_279 = torch.ops.aten.view.default(mul_710, [768, 1536, 1, 1]);  mul_710 = None
        convolution_130 = torch.ops.aten.convolution.default(mul_707, view_279, arg143_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_707 = view_279 = arg143_1 = None
        mul_711 = torch.ops.aten.mul.Tensor(convolution_130, 0.5)
        mul_712 = torch.ops.aten.mul.Tensor(convolution_130, 0.7071067811865476);  convolution_130 = None
        erf_84 = torch.ops.aten.erf.default(mul_712);  mul_712 = None
        add_196 = torch.ops.aten.add.Tensor(erf_84, 1);  erf_84 = None
        mul_713 = torch.ops.aten.mul.Tensor(mul_711, add_196);  mul_711 = add_196 = None
        mul_714 = torch.ops.aten.mul.Tensor(mul_713, 1.7015043497085571);  mul_713 = None
        view_280 = torch.ops.aten.view.default(arg144_1, [1, 768, -1]);  arg144_1 = None
        mul_715 = torch.ops.aten.mul.Tensor(arg145_1, 0.02946278254943948);  arg145_1 = None
        view_281 = torch.ops.aten.view.default(mul_715, [-1]);  mul_715 = None
        var_mean_93 = torch.ops.aten.var_mean.correction(view_280, [0, 2], correction = 0, keepdim = True)
        getitem_186 = var_mean_93[0]
        getitem_187 = var_mean_93[1];  var_mean_93 = None
        add_197 = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_93 = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_93 = torch.ops.aten.sub.Tensor(view_280, getitem_187);  view_280 = getitem_187 = None
        mul_716 = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = rsqrt_93 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(view_281, -1);  view_281 = None
        mul_717 = torch.ops.aten.mul.Tensor(mul_716, unsqueeze_93);  mul_716 = unsqueeze_93 = None
        view_282 = torch.ops.aten.view.default(mul_717, [768, 128, 3, 3]);  mul_717 = None
        convolution_131 = torch.ops.aten.convolution.default(mul_714, view_282, arg146_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_714 = view_282 = arg146_1 = None
        mul_718 = torch.ops.aten.mul.Tensor(convolution_131, 0.5)
        mul_719 = torch.ops.aten.mul.Tensor(convolution_131, 0.7071067811865476);  convolution_131 = None
        erf_85 = torch.ops.aten.erf.default(mul_719);  mul_719 = None
        add_198 = torch.ops.aten.add.Tensor(erf_85, 1);  erf_85 = None
        mul_720 = torch.ops.aten.mul.Tensor(mul_718, add_198);  mul_718 = add_198 = None
        mul_721 = torch.ops.aten.mul.Tensor(mul_720, 1.7015043497085571);  mul_720 = None
        view_283 = torch.ops.aten.view.default(arg147_1, [1, 768, -1]);  arg147_1 = None
        mul_722 = torch.ops.aten.mul.Tensor(arg148_1, 0.02946278254943948);  arg148_1 = None
        view_284 = torch.ops.aten.view.default(mul_722, [-1]);  mul_722 = None
        var_mean_94 = torch.ops.aten.var_mean.correction(view_283, [0, 2], correction = 0, keepdim = True)
        getitem_188 = var_mean_94[0]
        getitem_189 = var_mean_94[1];  var_mean_94 = None
        add_199 = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_94 = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        sub_94 = torch.ops.aten.sub.Tensor(view_283, getitem_189);  view_283 = getitem_189 = None
        mul_723 = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = rsqrt_94 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(view_284, -1);  view_284 = None
        mul_724 = torch.ops.aten.mul.Tensor(mul_723, unsqueeze_94);  mul_723 = unsqueeze_94 = None
        view_285 = torch.ops.aten.view.default(mul_724, [768, 128, 3, 3]);  mul_724 = None
        convolution_132 = torch.ops.aten.convolution.default(mul_721, view_285, arg149_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_721 = view_285 = arg149_1 = None
        mul_725 = torch.ops.aten.mul.Tensor(convolution_132, 0.5)
        mul_726 = torch.ops.aten.mul.Tensor(convolution_132, 0.7071067811865476);  convolution_132 = None
        erf_86 = torch.ops.aten.erf.default(mul_726);  mul_726 = None
        add_200 = torch.ops.aten.add.Tensor(erf_86, 1);  erf_86 = None
        mul_727 = torch.ops.aten.mul.Tensor(mul_725, add_200);  mul_725 = add_200 = None
        mul_728 = torch.ops.aten.mul.Tensor(mul_727, 1.7015043497085571);  mul_727 = None
        view_286 = torch.ops.aten.view.default(arg150_1, [1, 1536, -1]);  arg150_1 = None
        mul_729 = torch.ops.aten.mul.Tensor(arg151_1, 0.03608439182435161);  arg151_1 = None
        view_287 = torch.ops.aten.view.default(mul_729, [-1]);  mul_729 = None
        var_mean_95 = torch.ops.aten.var_mean.correction(view_286, [0, 2], correction = 0, keepdim = True)
        getitem_190 = var_mean_95[0]
        getitem_191 = var_mean_95[1];  var_mean_95 = None
        add_201 = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_95 = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        sub_95 = torch.ops.aten.sub.Tensor(view_286, getitem_191);  view_286 = getitem_191 = None
        mul_730 = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = rsqrt_95 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(view_287, -1);  view_287 = None
        mul_731 = torch.ops.aten.mul.Tensor(mul_730, unsqueeze_95);  mul_730 = unsqueeze_95 = None
        view_288 = torch.ops.aten.view.default(mul_731, [1536, 768, 1, 1]);  mul_731 = None
        convolution_133 = torch.ops.aten.convolution.default(mul_728, view_288, arg152_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_728 = view_288 = arg152_1 = None
        mean_20 = torch.ops.aten.mean.dim(convolution_133, [2, 3], True)
        convolution_134 = torch.ops.aten.convolution.default(mean_20, arg153_1, arg154_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_20 = arg153_1 = arg154_1 = None
        relu_19 = torch.ops.aten.relu.default(convolution_134);  convolution_134 = None
        convolution_135 = torch.ops.aten.convolution.default(relu_19, arg155_1, arg156_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_19 = arg155_1 = arg156_1 = None
        sigmoid_19 = torch.ops.aten.sigmoid.default(convolution_135);  convolution_135 = None
        mul_732 = torch.ops.aten.mul.Tensor(convolution_133, sigmoid_19);  convolution_133 = sigmoid_19 = None
        mul_733 = torch.ops.aten.mul.Tensor(mul_732, 2.0);  mul_732 = None
        mul_734 = torch.ops.aten.mul.Tensor(mul_733, arg157_1);  mul_733 = arg157_1 = None
        mul_735 = torch.ops.aten.mul.Tensor(mul_734, 0.2);  mul_734 = None
        add_202 = torch.ops.aten.add.Tensor(mul_735, add_193);  mul_735 = add_193 = None
        mul_736 = torch.ops.aten.mul.Tensor(add_202, 0.5)
        mul_737 = torch.ops.aten.mul.Tensor(add_202, 0.7071067811865476)
        erf_87 = torch.ops.aten.erf.default(mul_737);  mul_737 = None
        add_203 = torch.ops.aten.add.Tensor(erf_87, 1);  erf_87 = None
        mul_738 = torch.ops.aten.mul.Tensor(mul_736, add_203);  mul_736 = add_203 = None
        mul_739 = torch.ops.aten.mul.Tensor(mul_738, 1.7015043497085571);  mul_738 = None
        mul_740 = torch.ops.aten.mul.Tensor(mul_739, 0.9128709291752768);  mul_739 = None
        view_289 = torch.ops.aten.view.default(arg158_1, [1, 768, -1]);  arg158_1 = None
        mul_741 = torch.ops.aten.mul.Tensor(arg159_1, 0.02551551815399144);  arg159_1 = None
        view_290 = torch.ops.aten.view.default(mul_741, [-1]);  mul_741 = None
        var_mean_96 = torch.ops.aten.var_mean.correction(view_289, [0, 2], correction = 0, keepdim = True)
        getitem_192 = var_mean_96[0]
        getitem_193 = var_mean_96[1];  var_mean_96 = None
        add_204 = torch.ops.aten.add.Tensor(getitem_192, 1e-05);  getitem_192 = None
        rsqrt_96 = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        sub_96 = torch.ops.aten.sub.Tensor(view_289, getitem_193);  view_289 = getitem_193 = None
        mul_742 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = rsqrt_96 = None
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(view_290, -1);  view_290 = None
        mul_743 = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_96);  mul_742 = unsqueeze_96 = None
        view_291 = torch.ops.aten.view.default(mul_743, [768, 1536, 1, 1]);  mul_743 = None
        convolution_136 = torch.ops.aten.convolution.default(mul_740, view_291, arg160_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_740 = view_291 = arg160_1 = None
        mul_744 = torch.ops.aten.mul.Tensor(convolution_136, 0.5)
        mul_745 = torch.ops.aten.mul.Tensor(convolution_136, 0.7071067811865476);  convolution_136 = None
        erf_88 = torch.ops.aten.erf.default(mul_745);  mul_745 = None
        add_205 = torch.ops.aten.add.Tensor(erf_88, 1);  erf_88 = None
        mul_746 = torch.ops.aten.mul.Tensor(mul_744, add_205);  mul_744 = add_205 = None
        mul_747 = torch.ops.aten.mul.Tensor(mul_746, 1.7015043497085571);  mul_746 = None
        view_292 = torch.ops.aten.view.default(arg161_1, [1, 768, -1]);  arg161_1 = None
        mul_748 = torch.ops.aten.mul.Tensor(arg162_1, 0.02946278254943948);  arg162_1 = None
        view_293 = torch.ops.aten.view.default(mul_748, [-1]);  mul_748 = None
        var_mean_97 = torch.ops.aten.var_mean.correction(view_292, [0, 2], correction = 0, keepdim = True)
        getitem_194 = var_mean_97[0]
        getitem_195 = var_mean_97[1];  var_mean_97 = None
        add_206 = torch.ops.aten.add.Tensor(getitem_194, 1e-05);  getitem_194 = None
        rsqrt_97 = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        sub_97 = torch.ops.aten.sub.Tensor(view_292, getitem_195);  view_292 = getitem_195 = None
        mul_749 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = rsqrt_97 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(view_293, -1);  view_293 = None
        mul_750 = torch.ops.aten.mul.Tensor(mul_749, unsqueeze_97);  mul_749 = unsqueeze_97 = None
        view_294 = torch.ops.aten.view.default(mul_750, [768, 128, 3, 3]);  mul_750 = None
        convolution_137 = torch.ops.aten.convolution.default(mul_747, view_294, arg163_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_747 = view_294 = arg163_1 = None
        mul_751 = torch.ops.aten.mul.Tensor(convolution_137, 0.5)
        mul_752 = torch.ops.aten.mul.Tensor(convolution_137, 0.7071067811865476);  convolution_137 = None
        erf_89 = torch.ops.aten.erf.default(mul_752);  mul_752 = None
        add_207 = torch.ops.aten.add.Tensor(erf_89, 1);  erf_89 = None
        mul_753 = torch.ops.aten.mul.Tensor(mul_751, add_207);  mul_751 = add_207 = None
        mul_754 = torch.ops.aten.mul.Tensor(mul_753, 1.7015043497085571);  mul_753 = None
        view_295 = torch.ops.aten.view.default(arg164_1, [1, 768, -1]);  arg164_1 = None
        mul_755 = torch.ops.aten.mul.Tensor(arg165_1, 0.02946278254943948);  arg165_1 = None
        view_296 = torch.ops.aten.view.default(mul_755, [-1]);  mul_755 = None
        var_mean_98 = torch.ops.aten.var_mean.correction(view_295, [0, 2], correction = 0, keepdim = True)
        getitem_196 = var_mean_98[0]
        getitem_197 = var_mean_98[1];  var_mean_98 = None
        add_208 = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_98 = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
        sub_98 = torch.ops.aten.sub.Tensor(view_295, getitem_197);  view_295 = getitem_197 = None
        mul_756 = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = rsqrt_98 = None
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(view_296, -1);  view_296 = None
        mul_757 = torch.ops.aten.mul.Tensor(mul_756, unsqueeze_98);  mul_756 = unsqueeze_98 = None
        view_297 = torch.ops.aten.view.default(mul_757, [768, 128, 3, 3]);  mul_757 = None
        convolution_138 = torch.ops.aten.convolution.default(mul_754, view_297, arg166_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_754 = view_297 = arg166_1 = None
        mul_758 = torch.ops.aten.mul.Tensor(convolution_138, 0.5)
        mul_759 = torch.ops.aten.mul.Tensor(convolution_138, 0.7071067811865476);  convolution_138 = None
        erf_90 = torch.ops.aten.erf.default(mul_759);  mul_759 = None
        add_209 = torch.ops.aten.add.Tensor(erf_90, 1);  erf_90 = None
        mul_760 = torch.ops.aten.mul.Tensor(mul_758, add_209);  mul_758 = add_209 = None
        mul_761 = torch.ops.aten.mul.Tensor(mul_760, 1.7015043497085571);  mul_760 = None
        view_298 = torch.ops.aten.view.default(arg167_1, [1, 1536, -1]);  arg167_1 = None
        mul_762 = torch.ops.aten.mul.Tensor(arg168_1, 0.03608439182435161);  arg168_1 = None
        view_299 = torch.ops.aten.view.default(mul_762, [-1]);  mul_762 = None
        var_mean_99 = torch.ops.aten.var_mean.correction(view_298, [0, 2], correction = 0, keepdim = True)
        getitem_198 = var_mean_99[0]
        getitem_199 = var_mean_99[1];  var_mean_99 = None
        add_210 = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_99 = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        sub_99 = torch.ops.aten.sub.Tensor(view_298, getitem_199);  view_298 = getitem_199 = None
        mul_763 = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = rsqrt_99 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(view_299, -1);  view_299 = None
        mul_764 = torch.ops.aten.mul.Tensor(mul_763, unsqueeze_99);  mul_763 = unsqueeze_99 = None
        view_300 = torch.ops.aten.view.default(mul_764, [1536, 768, 1, 1]);  mul_764 = None
        convolution_139 = torch.ops.aten.convolution.default(mul_761, view_300, arg169_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_761 = view_300 = arg169_1 = None
        mean_21 = torch.ops.aten.mean.dim(convolution_139, [2, 3], True)
        convolution_140 = torch.ops.aten.convolution.default(mean_21, arg170_1, arg171_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_21 = arg170_1 = arg171_1 = None
        relu_20 = torch.ops.aten.relu.default(convolution_140);  convolution_140 = None
        convolution_141 = torch.ops.aten.convolution.default(relu_20, arg172_1, arg173_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = arg172_1 = arg173_1 = None
        sigmoid_20 = torch.ops.aten.sigmoid.default(convolution_141);  convolution_141 = None
        mul_765 = torch.ops.aten.mul.Tensor(convolution_139, sigmoid_20);  convolution_139 = sigmoid_20 = None
        mul_766 = torch.ops.aten.mul.Tensor(mul_765, 2.0);  mul_765 = None
        mul_767 = torch.ops.aten.mul.Tensor(mul_766, arg174_1);  mul_766 = arg174_1 = None
        mul_768 = torch.ops.aten.mul.Tensor(mul_767, 0.2);  mul_767 = None
        add_211 = torch.ops.aten.add.Tensor(mul_768, add_202);  mul_768 = add_202 = None
        mul_769 = torch.ops.aten.mul.Tensor(add_211, 0.5)
        mul_770 = torch.ops.aten.mul.Tensor(add_211, 0.7071067811865476);  add_211 = None
        erf_91 = torch.ops.aten.erf.default(mul_770);  mul_770 = None
        add_212 = torch.ops.aten.add.Tensor(erf_91, 1);  erf_91 = None
        mul_771 = torch.ops.aten.mul.Tensor(mul_769, add_212);  mul_769 = add_212 = None
        mul_772 = torch.ops.aten.mul.Tensor(mul_771, 1.7015043497085571);  mul_771 = None
        mul_773 = torch.ops.aten.mul.Tensor(mul_772, 0.8980265101338745);  mul_772 = None
        avg_pool2d_5 = torch.ops.aten.avg_pool2d.default(mul_773, [2, 2], [2, 2], [0, 0], True, False)
        view_301 = torch.ops.aten.view.default(arg175_1, [1, 1536, -1]);  arg175_1 = None
        mul_774 = torch.ops.aten.mul.Tensor(arg176_1, 0.02551551815399144);  arg176_1 = None
        view_302 = torch.ops.aten.view.default(mul_774, [-1]);  mul_774 = None
        var_mean_100 = torch.ops.aten.var_mean.correction(view_301, [0, 2], correction = 0, keepdim = True)
        getitem_200 = var_mean_100[0]
        getitem_201 = var_mean_100[1];  var_mean_100 = None
        add_213 = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_100 = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_100 = torch.ops.aten.sub.Tensor(view_301, getitem_201);  view_301 = getitem_201 = None
        mul_775 = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = rsqrt_100 = None
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(view_302, -1);  view_302 = None
        mul_776 = torch.ops.aten.mul.Tensor(mul_775, unsqueeze_100);  mul_775 = unsqueeze_100 = None
        view_303 = torch.ops.aten.view.default(mul_776, [1536, 1536, 1, 1]);  mul_776 = None
        convolution_142 = torch.ops.aten.convolution.default(avg_pool2d_5, view_303, arg177_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  avg_pool2d_5 = view_303 = arg177_1 = None
        view_304 = torch.ops.aten.view.default(arg178_1, [1, 768, -1]);  arg178_1 = None
        mul_777 = torch.ops.aten.mul.Tensor(arg179_1, 0.02551551815399144);  arg179_1 = None
        view_305 = torch.ops.aten.view.default(mul_777, [-1]);  mul_777 = None
        var_mean_101 = torch.ops.aten.var_mean.correction(view_304, [0, 2], correction = 0, keepdim = True)
        getitem_202 = var_mean_101[0]
        getitem_203 = var_mean_101[1];  var_mean_101 = None
        add_214 = torch.ops.aten.add.Tensor(getitem_202, 1e-05);  getitem_202 = None
        rsqrt_101 = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        sub_101 = torch.ops.aten.sub.Tensor(view_304, getitem_203);  view_304 = getitem_203 = None
        mul_778 = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = rsqrt_101 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(view_305, -1);  view_305 = None
        mul_779 = torch.ops.aten.mul.Tensor(mul_778, unsqueeze_101);  mul_778 = unsqueeze_101 = None
        view_306 = torch.ops.aten.view.default(mul_779, [768, 1536, 1, 1]);  mul_779 = None
        convolution_143 = torch.ops.aten.convolution.default(mul_773, view_306, arg180_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_773 = view_306 = arg180_1 = None
        mul_780 = torch.ops.aten.mul.Tensor(convolution_143, 0.5)
        mul_781 = torch.ops.aten.mul.Tensor(convolution_143, 0.7071067811865476);  convolution_143 = None
        erf_92 = torch.ops.aten.erf.default(mul_781);  mul_781 = None
        add_215 = torch.ops.aten.add.Tensor(erf_92, 1);  erf_92 = None
        mul_782 = torch.ops.aten.mul.Tensor(mul_780, add_215);  mul_780 = add_215 = None
        mul_783 = torch.ops.aten.mul.Tensor(mul_782, 1.7015043497085571);  mul_782 = None
        constant_pad_nd_9 = torch.ops.aten.constant_pad_nd.default(mul_783, [0, 1, 0, 1], 0.0);  mul_783 = None
        view_307 = torch.ops.aten.view.default(arg181_1, [1, 768, -1]);  arg181_1 = None
        mul_784 = torch.ops.aten.mul.Tensor(arg182_1, 0.02946278254943948);  arg182_1 = None
        view_308 = torch.ops.aten.view.default(mul_784, [-1]);  mul_784 = None
        var_mean_102 = torch.ops.aten.var_mean.correction(view_307, [0, 2], correction = 0, keepdim = True)
        getitem_204 = var_mean_102[0]
        getitem_205 = var_mean_102[1];  var_mean_102 = None
        add_216 = torch.ops.aten.add.Tensor(getitem_204, 1e-05);  getitem_204 = None
        rsqrt_102 = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_102 = torch.ops.aten.sub.Tensor(view_307, getitem_205);  view_307 = getitem_205 = None
        mul_785 = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = rsqrt_102 = None
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(view_308, -1);  view_308 = None
        mul_786 = torch.ops.aten.mul.Tensor(mul_785, unsqueeze_102);  mul_785 = unsqueeze_102 = None
        view_309 = torch.ops.aten.view.default(mul_786, [768, 128, 3, 3]);  mul_786 = None
        convolution_144 = torch.ops.aten.convolution.default(constant_pad_nd_9, view_309, arg183_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 6);  constant_pad_nd_9 = view_309 = arg183_1 = None
        mul_787 = torch.ops.aten.mul.Tensor(convolution_144, 0.5)
        mul_788 = torch.ops.aten.mul.Tensor(convolution_144, 0.7071067811865476);  convolution_144 = None
        erf_93 = torch.ops.aten.erf.default(mul_788);  mul_788 = None
        add_217 = torch.ops.aten.add.Tensor(erf_93, 1);  erf_93 = None
        mul_789 = torch.ops.aten.mul.Tensor(mul_787, add_217);  mul_787 = add_217 = None
        mul_790 = torch.ops.aten.mul.Tensor(mul_789, 1.7015043497085571);  mul_789 = None
        view_310 = torch.ops.aten.view.default(arg184_1, [1, 768, -1]);  arg184_1 = None
        mul_791 = torch.ops.aten.mul.Tensor(arg185_1, 0.02946278254943948);  arg185_1 = None
        view_311 = torch.ops.aten.view.default(mul_791, [-1]);  mul_791 = None
        var_mean_103 = torch.ops.aten.var_mean.correction(view_310, [0, 2], correction = 0, keepdim = True)
        getitem_206 = var_mean_103[0]
        getitem_207 = var_mean_103[1];  var_mean_103 = None
        add_218 = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_103 = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_103 = torch.ops.aten.sub.Tensor(view_310, getitem_207);  view_310 = getitem_207 = None
        mul_792 = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = rsqrt_103 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(view_311, -1);  view_311 = None
        mul_793 = torch.ops.aten.mul.Tensor(mul_792, unsqueeze_103);  mul_792 = unsqueeze_103 = None
        view_312 = torch.ops.aten.view.default(mul_793, [768, 128, 3, 3]);  mul_793 = None
        convolution_145 = torch.ops.aten.convolution.default(mul_790, view_312, arg186_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_790 = view_312 = arg186_1 = None
        mul_794 = torch.ops.aten.mul.Tensor(convolution_145, 0.5)
        mul_795 = torch.ops.aten.mul.Tensor(convolution_145, 0.7071067811865476);  convolution_145 = None
        erf_94 = torch.ops.aten.erf.default(mul_795);  mul_795 = None
        add_219 = torch.ops.aten.add.Tensor(erf_94, 1);  erf_94 = None
        mul_796 = torch.ops.aten.mul.Tensor(mul_794, add_219);  mul_794 = add_219 = None
        mul_797 = torch.ops.aten.mul.Tensor(mul_796, 1.7015043497085571);  mul_796 = None
        view_313 = torch.ops.aten.view.default(arg187_1, [1, 1536, -1]);  arg187_1 = None
        mul_798 = torch.ops.aten.mul.Tensor(arg188_1, 0.03608439182435161);  arg188_1 = None
        view_314 = torch.ops.aten.view.default(mul_798, [-1]);  mul_798 = None
        var_mean_104 = torch.ops.aten.var_mean.correction(view_313, [0, 2], correction = 0, keepdim = True)
        getitem_208 = var_mean_104[0]
        getitem_209 = var_mean_104[1];  var_mean_104 = None
        add_220 = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_104 = torch.ops.aten.rsqrt.default(add_220);  add_220 = None
        sub_104 = torch.ops.aten.sub.Tensor(view_313, getitem_209);  view_313 = getitem_209 = None
        mul_799 = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = rsqrt_104 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(view_314, -1);  view_314 = None
        mul_800 = torch.ops.aten.mul.Tensor(mul_799, unsqueeze_104);  mul_799 = unsqueeze_104 = None
        view_315 = torch.ops.aten.view.default(mul_800, [1536, 768, 1, 1]);  mul_800 = None
        convolution_146 = torch.ops.aten.convolution.default(mul_797, view_315, arg189_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_797 = view_315 = arg189_1 = None
        mean_22 = torch.ops.aten.mean.dim(convolution_146, [2, 3], True)
        convolution_147 = torch.ops.aten.convolution.default(mean_22, arg190_1, arg191_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_22 = arg190_1 = arg191_1 = None
        relu_21 = torch.ops.aten.relu.default(convolution_147);  convolution_147 = None
        convolution_148 = torch.ops.aten.convolution.default(relu_21, arg192_1, arg193_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_21 = arg192_1 = arg193_1 = None
        sigmoid_21 = torch.ops.aten.sigmoid.default(convolution_148);  convolution_148 = None
        mul_801 = torch.ops.aten.mul.Tensor(convolution_146, sigmoid_21);  convolution_146 = sigmoid_21 = None
        mul_802 = torch.ops.aten.mul.Tensor(mul_801, 2.0);  mul_801 = None
        mul_803 = torch.ops.aten.mul.Tensor(mul_802, arg194_1);  mul_802 = arg194_1 = None
        mul_804 = torch.ops.aten.mul.Tensor(mul_803, 0.2);  mul_803 = None
        add_221 = torch.ops.aten.add.Tensor(mul_804, convolution_142);  mul_804 = convolution_142 = None
        mul_805 = torch.ops.aten.mul.Tensor(add_221, 0.5)
        mul_806 = torch.ops.aten.mul.Tensor(add_221, 0.7071067811865476)
        erf_95 = torch.ops.aten.erf.default(mul_806);  mul_806 = None
        add_222 = torch.ops.aten.add.Tensor(erf_95, 1);  erf_95 = None
        mul_807 = torch.ops.aten.mul.Tensor(mul_805, add_222);  mul_805 = add_222 = None
        mul_808 = torch.ops.aten.mul.Tensor(mul_807, 1.7015043497085571);  mul_807 = None
        mul_809 = torch.ops.aten.mul.Tensor(mul_808, 0.9805806756909201);  mul_808 = None
        view_316 = torch.ops.aten.view.default(arg195_1, [1, 768, -1]);  arg195_1 = None
        mul_810 = torch.ops.aten.mul.Tensor(arg196_1, 0.02551551815399144);  arg196_1 = None
        view_317 = torch.ops.aten.view.default(mul_810, [-1]);  mul_810 = None
        var_mean_105 = torch.ops.aten.var_mean.correction(view_316, [0, 2], correction = 0, keepdim = True)
        getitem_210 = var_mean_105[0]
        getitem_211 = var_mean_105[1];  var_mean_105 = None
        add_223 = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_105 = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_105 = torch.ops.aten.sub.Tensor(view_316, getitem_211);  view_316 = getitem_211 = None
        mul_811 = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = rsqrt_105 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(view_317, -1);  view_317 = None
        mul_812 = torch.ops.aten.mul.Tensor(mul_811, unsqueeze_105);  mul_811 = unsqueeze_105 = None
        view_318 = torch.ops.aten.view.default(mul_812, [768, 1536, 1, 1]);  mul_812 = None
        convolution_149 = torch.ops.aten.convolution.default(mul_809, view_318, arg197_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_809 = view_318 = arg197_1 = None
        mul_813 = torch.ops.aten.mul.Tensor(convolution_149, 0.5)
        mul_814 = torch.ops.aten.mul.Tensor(convolution_149, 0.7071067811865476);  convolution_149 = None
        erf_96 = torch.ops.aten.erf.default(mul_814);  mul_814 = None
        add_224 = torch.ops.aten.add.Tensor(erf_96, 1);  erf_96 = None
        mul_815 = torch.ops.aten.mul.Tensor(mul_813, add_224);  mul_813 = add_224 = None
        mul_816 = torch.ops.aten.mul.Tensor(mul_815, 1.7015043497085571);  mul_815 = None
        view_319 = torch.ops.aten.view.default(arg198_1, [1, 768, -1]);  arg198_1 = None
        mul_817 = torch.ops.aten.mul.Tensor(arg199_1, 0.02946278254943948);  arg199_1 = None
        view_320 = torch.ops.aten.view.default(mul_817, [-1]);  mul_817 = None
        var_mean_106 = torch.ops.aten.var_mean.correction(view_319, [0, 2], correction = 0, keepdim = True)
        getitem_212 = var_mean_106[0]
        getitem_213 = var_mean_106[1];  var_mean_106 = None
        add_225 = torch.ops.aten.add.Tensor(getitem_212, 1e-05);  getitem_212 = None
        rsqrt_106 = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        sub_106 = torch.ops.aten.sub.Tensor(view_319, getitem_213);  view_319 = getitem_213 = None
        mul_818 = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = rsqrt_106 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(view_320, -1);  view_320 = None
        mul_819 = torch.ops.aten.mul.Tensor(mul_818, unsqueeze_106);  mul_818 = unsqueeze_106 = None
        view_321 = torch.ops.aten.view.default(mul_819, [768, 128, 3, 3]);  mul_819 = None
        convolution_150 = torch.ops.aten.convolution.default(mul_816, view_321, arg200_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_816 = view_321 = arg200_1 = None
        mul_820 = torch.ops.aten.mul.Tensor(convolution_150, 0.5)
        mul_821 = torch.ops.aten.mul.Tensor(convolution_150, 0.7071067811865476);  convolution_150 = None
        erf_97 = torch.ops.aten.erf.default(mul_821);  mul_821 = None
        add_226 = torch.ops.aten.add.Tensor(erf_97, 1);  erf_97 = None
        mul_822 = torch.ops.aten.mul.Tensor(mul_820, add_226);  mul_820 = add_226 = None
        mul_823 = torch.ops.aten.mul.Tensor(mul_822, 1.7015043497085571);  mul_822 = None
        view_322 = torch.ops.aten.view.default(arg201_1, [1, 768, -1]);  arg201_1 = None
        mul_824 = torch.ops.aten.mul.Tensor(arg202_1, 0.02946278254943948);  arg202_1 = None
        view_323 = torch.ops.aten.view.default(mul_824, [-1]);  mul_824 = None
        var_mean_107 = torch.ops.aten.var_mean.correction(view_322, [0, 2], correction = 0, keepdim = True)
        getitem_214 = var_mean_107[0]
        getitem_215 = var_mean_107[1];  var_mean_107 = None
        add_227 = torch.ops.aten.add.Tensor(getitem_214, 1e-05);  getitem_214 = None
        rsqrt_107 = torch.ops.aten.rsqrt.default(add_227);  add_227 = None
        sub_107 = torch.ops.aten.sub.Tensor(view_322, getitem_215);  view_322 = getitem_215 = None
        mul_825 = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = rsqrt_107 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(view_323, -1);  view_323 = None
        mul_826 = torch.ops.aten.mul.Tensor(mul_825, unsqueeze_107);  mul_825 = unsqueeze_107 = None
        view_324 = torch.ops.aten.view.default(mul_826, [768, 128, 3, 3]);  mul_826 = None
        convolution_151 = torch.ops.aten.convolution.default(mul_823, view_324, arg203_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_823 = view_324 = arg203_1 = None
        mul_827 = torch.ops.aten.mul.Tensor(convolution_151, 0.5)
        mul_828 = torch.ops.aten.mul.Tensor(convolution_151, 0.7071067811865476);  convolution_151 = None
        erf_98 = torch.ops.aten.erf.default(mul_828);  mul_828 = None
        add_228 = torch.ops.aten.add.Tensor(erf_98, 1);  erf_98 = None
        mul_829 = torch.ops.aten.mul.Tensor(mul_827, add_228);  mul_827 = add_228 = None
        mul_830 = torch.ops.aten.mul.Tensor(mul_829, 1.7015043497085571);  mul_829 = None
        view_325 = torch.ops.aten.view.default(arg204_1, [1, 1536, -1]);  arg204_1 = None
        mul_831 = torch.ops.aten.mul.Tensor(arg205_1, 0.03608439182435161);  arg205_1 = None
        view_326 = torch.ops.aten.view.default(mul_831, [-1]);  mul_831 = None
        var_mean_108 = torch.ops.aten.var_mean.correction(view_325, [0, 2], correction = 0, keepdim = True)
        getitem_216 = var_mean_108[0]
        getitem_217 = var_mean_108[1];  var_mean_108 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_108 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_108 = torch.ops.aten.sub.Tensor(view_325, getitem_217);  view_325 = getitem_217 = None
        mul_832 = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = rsqrt_108 = None
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(view_326, -1);  view_326 = None
        mul_833 = torch.ops.aten.mul.Tensor(mul_832, unsqueeze_108);  mul_832 = unsqueeze_108 = None
        view_327 = torch.ops.aten.view.default(mul_833, [1536, 768, 1, 1]);  mul_833 = None
        convolution_152 = torch.ops.aten.convolution.default(mul_830, view_327, arg206_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_830 = view_327 = arg206_1 = None
        mean_23 = torch.ops.aten.mean.dim(convolution_152, [2, 3], True)
        convolution_153 = torch.ops.aten.convolution.default(mean_23, arg207_1, arg208_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_23 = arg207_1 = arg208_1 = None
        relu_22 = torch.ops.aten.relu.default(convolution_153);  convolution_153 = None
        convolution_154 = torch.ops.aten.convolution.default(relu_22, arg209_1, arg210_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_22 = arg209_1 = arg210_1 = None
        sigmoid_22 = torch.ops.aten.sigmoid.default(convolution_154);  convolution_154 = None
        mul_834 = torch.ops.aten.mul.Tensor(convolution_152, sigmoid_22);  convolution_152 = sigmoid_22 = None
        mul_835 = torch.ops.aten.mul.Tensor(mul_834, 2.0);  mul_834 = None
        mul_836 = torch.ops.aten.mul.Tensor(mul_835, arg211_1);  mul_835 = arg211_1 = None
        mul_837 = torch.ops.aten.mul.Tensor(mul_836, 0.2);  mul_836 = None
        add_230 = torch.ops.aten.add.Tensor(mul_837, add_221);  mul_837 = add_221 = None
        mul_838 = torch.ops.aten.mul.Tensor(add_230, 0.5)
        mul_839 = torch.ops.aten.mul.Tensor(add_230, 0.7071067811865476)
        erf_99 = torch.ops.aten.erf.default(mul_839);  mul_839 = None
        add_231 = torch.ops.aten.add.Tensor(erf_99, 1);  erf_99 = None
        mul_840 = torch.ops.aten.mul.Tensor(mul_838, add_231);  mul_838 = add_231 = None
        mul_841 = torch.ops.aten.mul.Tensor(mul_840, 1.7015043497085571);  mul_840 = None
        mul_842 = torch.ops.aten.mul.Tensor(mul_841, 0.9622504486493761);  mul_841 = None
        view_328 = torch.ops.aten.view.default(arg212_1, [1, 768, -1]);  arg212_1 = None
        mul_843 = torch.ops.aten.mul.Tensor(arg213_1, 0.02551551815399144);  arg213_1 = None
        view_329 = torch.ops.aten.view.default(mul_843, [-1]);  mul_843 = None
        var_mean_109 = torch.ops.aten.var_mean.correction(view_328, [0, 2], correction = 0, keepdim = True)
        getitem_218 = var_mean_109[0]
        getitem_219 = var_mean_109[1];  var_mean_109 = None
        add_232 = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
        rsqrt_109 = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        sub_109 = torch.ops.aten.sub.Tensor(view_328, getitem_219);  view_328 = getitem_219 = None
        mul_844 = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = rsqrt_109 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(view_329, -1);  view_329 = None
        mul_845 = torch.ops.aten.mul.Tensor(mul_844, unsqueeze_109);  mul_844 = unsqueeze_109 = None
        view_330 = torch.ops.aten.view.default(mul_845, [768, 1536, 1, 1]);  mul_845 = None
        convolution_155 = torch.ops.aten.convolution.default(mul_842, view_330, arg214_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_842 = view_330 = arg214_1 = None
        mul_846 = torch.ops.aten.mul.Tensor(convolution_155, 0.5)
        mul_847 = torch.ops.aten.mul.Tensor(convolution_155, 0.7071067811865476);  convolution_155 = None
        erf_100 = torch.ops.aten.erf.default(mul_847);  mul_847 = None
        add_233 = torch.ops.aten.add.Tensor(erf_100, 1);  erf_100 = None
        mul_848 = torch.ops.aten.mul.Tensor(mul_846, add_233);  mul_846 = add_233 = None
        mul_849 = torch.ops.aten.mul.Tensor(mul_848, 1.7015043497085571);  mul_848 = None
        view_331 = torch.ops.aten.view.default(arg215_1, [1, 768, -1]);  arg215_1 = None
        mul_850 = torch.ops.aten.mul.Tensor(arg216_1, 0.02946278254943948);  arg216_1 = None
        view_332 = torch.ops.aten.view.default(mul_850, [-1]);  mul_850 = None
        var_mean_110 = torch.ops.aten.var_mean.correction(view_331, [0, 2], correction = 0, keepdim = True)
        getitem_220 = var_mean_110[0]
        getitem_221 = var_mean_110[1];  var_mean_110 = None
        add_234 = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_110 = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_110 = torch.ops.aten.sub.Tensor(view_331, getitem_221);  view_331 = getitem_221 = None
        mul_851 = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = rsqrt_110 = None
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(view_332, -1);  view_332 = None
        mul_852 = torch.ops.aten.mul.Tensor(mul_851, unsqueeze_110);  mul_851 = unsqueeze_110 = None
        view_333 = torch.ops.aten.view.default(mul_852, [768, 128, 3, 3]);  mul_852 = None
        convolution_156 = torch.ops.aten.convolution.default(mul_849, view_333, arg217_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_849 = view_333 = arg217_1 = None
        mul_853 = torch.ops.aten.mul.Tensor(convolution_156, 0.5)
        mul_854 = torch.ops.aten.mul.Tensor(convolution_156, 0.7071067811865476);  convolution_156 = None
        erf_101 = torch.ops.aten.erf.default(mul_854);  mul_854 = None
        add_235 = torch.ops.aten.add.Tensor(erf_101, 1);  erf_101 = None
        mul_855 = torch.ops.aten.mul.Tensor(mul_853, add_235);  mul_853 = add_235 = None
        mul_856 = torch.ops.aten.mul.Tensor(mul_855, 1.7015043497085571);  mul_855 = None
        view_334 = torch.ops.aten.view.default(arg218_1, [1, 768, -1]);  arg218_1 = None
        mul_857 = torch.ops.aten.mul.Tensor(arg219_1, 0.02946278254943948);  arg219_1 = None
        view_335 = torch.ops.aten.view.default(mul_857, [-1]);  mul_857 = None
        var_mean_111 = torch.ops.aten.var_mean.correction(view_334, [0, 2], correction = 0, keepdim = True)
        getitem_222 = var_mean_111[0]
        getitem_223 = var_mean_111[1];  var_mean_111 = None
        add_236 = torch.ops.aten.add.Tensor(getitem_222, 1e-05);  getitem_222 = None
        rsqrt_111 = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_111 = torch.ops.aten.sub.Tensor(view_334, getitem_223);  view_334 = getitem_223 = None
        mul_858 = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = rsqrt_111 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(view_335, -1);  view_335 = None
        mul_859 = torch.ops.aten.mul.Tensor(mul_858, unsqueeze_111);  mul_858 = unsqueeze_111 = None
        view_336 = torch.ops.aten.view.default(mul_859, [768, 128, 3, 3]);  mul_859 = None
        convolution_157 = torch.ops.aten.convolution.default(mul_856, view_336, arg220_1, [1, 1], [1, 1], [1, 1], False, [0, 0], 6);  mul_856 = view_336 = arg220_1 = None
        mul_860 = torch.ops.aten.mul.Tensor(convolution_157, 0.5)
        mul_861 = torch.ops.aten.mul.Tensor(convolution_157, 0.7071067811865476);  convolution_157 = None
        erf_102 = torch.ops.aten.erf.default(mul_861);  mul_861 = None
        add_237 = torch.ops.aten.add.Tensor(erf_102, 1);  erf_102 = None
        mul_862 = torch.ops.aten.mul.Tensor(mul_860, add_237);  mul_860 = add_237 = None
        mul_863 = torch.ops.aten.mul.Tensor(mul_862, 1.7015043497085571);  mul_862 = None
        view_337 = torch.ops.aten.view.default(arg221_1, [1, 1536, -1]);  arg221_1 = None
        mul_864 = torch.ops.aten.mul.Tensor(arg222_1, 0.03608439182435161);  arg222_1 = None
        view_338 = torch.ops.aten.view.default(mul_864, [-1]);  mul_864 = None
        var_mean_112 = torch.ops.aten.var_mean.correction(view_337, [0, 2], correction = 0, keepdim = True)
        getitem_224 = var_mean_112[0]
        getitem_225 = var_mean_112[1];  var_mean_112 = None
        add_238 = torch.ops.aten.add.Tensor(getitem_224, 1e-05);  getitem_224 = None
        rsqrt_112 = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
        sub_112 = torch.ops.aten.sub.Tensor(view_337, getitem_225);  view_337 = getitem_225 = None
        mul_865 = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = rsqrt_112 = None
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(view_338, -1);  view_338 = None
        mul_866 = torch.ops.aten.mul.Tensor(mul_865, unsqueeze_112);  mul_865 = unsqueeze_112 = None
        view_339 = torch.ops.aten.view.default(mul_866, [1536, 768, 1, 1]);  mul_866 = None
        convolution_158 = torch.ops.aten.convolution.default(mul_863, view_339, arg223_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_863 = view_339 = arg223_1 = None
        mean_24 = torch.ops.aten.mean.dim(convolution_158, [2, 3], True)
        convolution_159 = torch.ops.aten.convolution.default(mean_24, arg224_1, arg225_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_24 = arg224_1 = arg225_1 = None
        relu_23 = torch.ops.aten.relu.default(convolution_159);  convolution_159 = None
        convolution_160 = torch.ops.aten.convolution.default(relu_23, arg226_1, arg227_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg226_1 = arg227_1 = None
        sigmoid_23 = torch.ops.aten.sigmoid.default(convolution_160);  convolution_160 = None
        mul_867 = torch.ops.aten.mul.Tensor(convolution_158, sigmoid_23);  convolution_158 = sigmoid_23 = None
        mul_868 = torch.ops.aten.mul.Tensor(mul_867, 2.0);  mul_867 = None
        mul_869 = torch.ops.aten.mul.Tensor(mul_868, arg228_1);  mul_868 = arg228_1 = None
        mul_870 = torch.ops.aten.mul.Tensor(mul_869, 0.2);  mul_869 = None
        add_239 = torch.ops.aten.add.Tensor(mul_870, add_230);  mul_870 = add_230 = None
        view_340 = torch.ops.aten.view.default(arg229_1, [1, 3072, -1]);  arg229_1 = None
        mul_871 = torch.ops.aten.mul.Tensor(arg230_1, 0.02551551815399144);  arg230_1 = None
        view_341 = torch.ops.aten.view.default(mul_871, [-1]);  mul_871 = None
        var_mean_113 = torch.ops.aten.var_mean.correction(view_340, [0, 2], correction = 0, keepdim = True)
        getitem_226 = var_mean_113[0]
        getitem_227 = var_mean_113[1];  var_mean_113 = None
        add_240 = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_113 = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_113 = torch.ops.aten.sub.Tensor(view_340, getitem_227);  view_340 = getitem_227 = None
        mul_872 = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = rsqrt_113 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(view_341, -1);  view_341 = None
        mul_873 = torch.ops.aten.mul.Tensor(mul_872, unsqueeze_113);  mul_872 = unsqueeze_113 = None
        view_342 = torch.ops.aten.view.default(mul_873, [3072, 1536, 1, 1]);  mul_873 = None
        convolution_161 = torch.ops.aten.convolution.default(add_239, view_342, arg231_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_239 = view_342 = arg231_1 = None
        mul_874 = torch.ops.aten.mul.Tensor(convolution_161, 0.5)
        mul_875 = torch.ops.aten.mul.Tensor(convolution_161, 0.7071067811865476);  convolution_161 = None
        erf_103 = torch.ops.aten.erf.default(mul_875);  mul_875 = None
        add_241 = torch.ops.aten.add.Tensor(erf_103, 1);  erf_103 = None
        mul_876 = torch.ops.aten.mul.Tensor(mul_874, add_241);  mul_874 = add_241 = None
        mul_877 = torch.ops.aten.mul.Tensor(mul_876, 1.7015043497085571);  mul_876 = None
        mean_25 = torch.ops.aten.mean.dim(mul_877, [-1, -2], True);  mul_877 = None
        view_343 = torch.ops.aten.view.default(mean_25, [8, 3072]);  mean_25 = None
        permute_1 = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg233_1, view_343, permute_1);  arg233_1 = view_343 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 256, 256), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf1, (16, 3, 3, 3), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf2, (16, 1, 1, 1), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf3, (16,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf4, (32, 16, 3, 3), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf5, (32, 1, 1, 1), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf6, (32,), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64, 32, 3, 3), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64, 1, 1, 1), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf10, (128, 64, 3, 3), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (128, 1, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf12, (128,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf13, (256, 128, 1, 1), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf14, (256, 1, 1, 1), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf15, (256,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf16, (128, 128, 1, 1), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf17, (128, 1, 1, 1), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf18, (128,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf19, (128, 128, 3, 3), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf20, (128, 1, 1, 1), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf22, (128, 128, 3, 3), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf23, (128, 1, 1, 1), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf24, (128,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256, 128, 1, 1), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf26, (256, 1, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128, 256, 1, 1), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256, 128, 1, 1), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf32, (), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf33, (512, 256, 1, 1), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf34, (512, 1, 1, 1), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf35, (512,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256, 256, 1, 1), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256, 1, 1, 1), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256, 128, 3, 3), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256, 1, 1, 1), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256, 128, 3, 3), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256, 1, 1, 1), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf45, (512, 256, 1, 1), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf46, (512, 1, 1, 1), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf47, (512,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf48, (256, 512, 1, 1), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf49, (256,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512, 256, 1, 1), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf51, (512,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf52, (), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf53, (256, 512, 1, 1), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf54, (256, 1, 1, 1), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256, 128, 3, 3), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256, 1, 1, 1), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf59, (256, 128, 3, 3), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf60, (256, 1, 1, 1), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf61, (256,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf62, (512, 256, 1, 1), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf63, (512, 1, 1, 1), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf64, (512,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf65, (256, 512, 1, 1), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf66, (256,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512, 256, 1, 1), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf69, (), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 3145728, device=device(type='cuda', index=0))
    reader.tensor(buf70, (1536, 512, 1, 1), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf71, (1536, 1, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf72, (1536,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1572864, device=device(type='cuda', index=0))
    reader.tensor(buf73, (768, 512, 1, 1), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf74, (768, 1, 1, 1), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf75, (768,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf76, (768, 128, 3, 3), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf77, (768, 1, 1, 1), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf78, (768,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf79, (768, 128, 3, 3), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf80, (768, 1, 1, 1), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf81, (768,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf82, (1536, 768, 1, 1), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf83, (1536, 1, 1, 1), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1536,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf85, (768, 1536, 1, 1), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf86, (768,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf87, (1536, 768, 1, 1), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf88, (1536,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf89, (), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf90, (768, 1536, 1, 1), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf91, (768, 1, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf92, (768,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf93, (768, 128, 3, 3), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf94, (768, 1, 1, 1), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf95, (768,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf96, (768, 128, 3, 3), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf97, (768, 1, 1, 1), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf98, (768,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf99, (1536, 768, 1, 1), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1536, 1, 1, 1), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1536,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf102, (768, 1536, 1, 1), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf103, (768,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf104, (1536, 768, 1, 1), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1536,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf106, (), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf107, (768, 1536, 1, 1), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf108, (768, 1, 1, 1), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf109, (768,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf110, (768, 128, 3, 3), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf111, (768, 1, 1, 1), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf112, (768,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf113, (768, 128, 3, 3), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf114, (768, 1, 1, 1), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf115, (768,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf116, (1536, 768, 1, 1), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf117, (1536, 1, 1, 1), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf118, (1536,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768, 1536, 1, 1), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf121, (1536, 768, 1, 1), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf122, (1536,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf123, (), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768, 1536, 1, 1), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768, 1, 1, 1), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768, 128, 3, 3), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf128, (768, 1, 1, 1), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768, 128, 3, 3), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768, 1, 1, 1), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf133, (1536, 768, 1, 1), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf134, (1536, 1, 1, 1), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf135, (1536,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768, 1536, 1, 1), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf138, (1536, 768, 1, 1), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf139, (1536,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf140, (), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768, 1536, 1, 1), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768, 1, 1, 1), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768, 128, 3, 3), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768, 1, 1, 1), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf146, (768,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768, 128, 3, 3), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768, 1, 1, 1), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf150, (1536, 768, 1, 1), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf151, (1536, 1, 1, 1), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf152, (1536,), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768, 1536, 1, 1), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf155, (1536, 768, 1, 1), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf156, (1536,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf157, (), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf158, (768, 1536, 1, 1), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768, 1, 1, 1), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768, 128, 3, 3), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768, 1, 1, 1), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf164, (768, 128, 3, 3), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768, 1, 1, 1), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf167, (1536, 768, 1, 1), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf168, (1536, 1, 1, 1), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf169, (1536,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf170, (768, 1536, 1, 1), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf171, (768,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf172, (1536, 768, 1, 1), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf173, (1536,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf174, (), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf175, (1536, 1536, 1, 1), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf176, (1536, 1, 1, 1), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf177, (1536,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf178, (768, 1536, 1, 1), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf179, (768, 1, 1, 1), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf180, (768,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf181, (768, 128, 3, 3), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf182, (768, 1, 1, 1), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf183, (768,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf184, (768, 128, 3, 3), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf185, (768, 1, 1, 1), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf186, (768,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf187, (1536, 768, 1, 1), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf188, (1536, 1, 1, 1), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf189, (1536,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf190, (768, 1536, 1, 1), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf191, (768,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf192, (1536, 768, 1, 1), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf193, (1536,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf194, (), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf195, (768, 1536, 1, 1), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf196, (768, 1, 1, 1), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf197, (768,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf198, (768, 128, 3, 3), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf199, (768, 1, 1, 1), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf200, (768,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf201, (768, 128, 3, 3), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf202, (768, 1, 1, 1), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf203, (768,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf204, (1536, 768, 1, 1), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf205, (1536, 1, 1, 1), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf206, (1536,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf207, (768, 1536, 1, 1), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf208, (768,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf209, (1536, 768, 1, 1), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf210, (1536,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf211, (), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf212, (768, 1536, 1, 1), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf213, (768, 1, 1, 1), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf214, (768,), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf215, (768, 128, 3, 3), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf216, (768, 1, 1, 1), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf217, (768,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 3538944, device=device(type='cuda', index=0))
    reader.tensor(buf218, (768, 128, 3, 3), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf219, (768, 1, 1, 1), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf220, (768,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf221, (1536, 768, 1, 1), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf222, (1536, 1, 1, 1), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf223, (1536,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf224, (768, 1536, 1, 1), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf225, (768,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf226, (1536, 768, 1, 1), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 6144, device=device(type='cuda', index=0))
    reader.tensor(buf227, (1536,), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 4, device=device(type='cuda', index=0))
    reader.tensor(buf228, (), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 18874368, device=device(type='cuda', index=0))
    reader.tensor(buf229, (3072, 1536, 1, 1), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf230, (3072, 1, 1, 1), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 12288, device=device(type='cuda', index=0))
    reader.tensor(buf231, (3072,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 12288000, device=device(type='cuda', index=0))
    reader.tensor(buf232, (1000, 3072), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf233, (1000,), is_leaf=True)  # arg233_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)