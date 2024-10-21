
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1):
        convolution_57 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_104 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_28 = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_28 = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_158 = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(mul_158, -1);  mul_158 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        sub_36 = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_225);  convolution_57 = unsqueeze_225 = None
        mul_159 = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_227);  sub_36 = unsqueeze_227 = None
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_160 = torch.ops.aten.mul.Tensor(mul_159, unsqueeze_229);  mul_159 = unsqueeze_229 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_105 = torch.ops.aten.add.Tensor(mul_160, unsqueeze_231);  mul_160 = unsqueeze_231 = None
        relu_1 = torch.ops.aten.relu.default(add_105);  add_105 = None
        convolution_58 = torch.ops.aten.convolution.default(relu_1, arg6_1, arg7_1, [4, 4], [0, 0], [1, 1], False, [0, 0], 1);  relu_1 = arg6_1 = arg7_1 = None
        add_106 = torch.ops.aten.add.Tensor(arg9_1, 1e-05);  arg9_1 = None
        sqrt_29 = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_29 = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_161 = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(arg8_1, -1);  arg8_1 = None
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(mul_161, -1);  mul_161 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        sub_37 = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_233);  convolution_58 = unsqueeze_233 = None
        mul_162 = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_235);  sub_37 = unsqueeze_235 = None
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_163 = torch.ops.aten.mul.Tensor(mul_162, unsqueeze_237);  mul_162 = unsqueeze_237 = None
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(arg11_1, -1);  arg11_1 = None
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_107 = torch.ops.aten.add.Tensor(mul_163, unsqueeze_239);  mul_163 = unsqueeze_239 = None
        add_108 = torch.ops.aten.add.Tensor(add_107, arg12_1);  add_107 = arg12_1 = None
        add_109 = torch.ops.aten.add.Tensor(arg14_1, 1e-05);  arg14_1 = None
        sqrt_30 = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_30 = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_164 = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(arg13_1, -1);  arg13_1 = None
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(mul_164, -1);  mul_164 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        sub_38 = torch.ops.aten.sub.Tensor(add_108, unsqueeze_241);  unsqueeze_241 = None
        mul_165 = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_243);  sub_38 = unsqueeze_243 = None
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_166 = torch.ops.aten.mul.Tensor(mul_165, unsqueeze_245);  mul_165 = unsqueeze_245 = None
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(arg16_1, -1);  arg16_1 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_110 = torch.ops.aten.add.Tensor(mul_166, unsqueeze_247);  mul_166 = unsqueeze_247 = None
        convolution_59 = torch.ops.aten.convolution.default(add_110, arg17_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_110 = arg17_1 = None
        mul_167 = torch.ops.aten.mul.Tensor(convolution_59, 0.5)
        mul_168 = torch.ops.aten.mul.Tensor(convolution_59, 0.7071067811865476);  convolution_59 = None
        erf_22 = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_111 = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_167, add_111);  mul_167 = add_111 = None
        convolution_60 = torch.ops.aten.convolution.default(mul_169, arg18_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_169 = arg18_1 = None
        mul_170 = torch.ops.aten.mul.Tensor(convolution_60, 0.5)
        mul_171 = torch.ops.aten.mul.Tensor(convolution_60, 0.7071067811865476);  convolution_60 = None
        erf_23 = torch.ops.aten.erf.default(mul_171);  mul_171 = None
        add_112 = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_172 = torch.ops.aten.mul.Tensor(mul_170, add_112);  mul_170 = add_112 = None
        convolution_61 = torch.ops.aten.convolution.default(mul_172, arg19_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_172 = arg19_1 = None
        add_113 = torch.ops.aten.add.Tensor(add_108, convolution_61);  add_108 = convolution_61 = None
        add_114 = torch.ops.aten.add.Tensor(arg21_1, 1e-05);  arg21_1 = None
        sqrt_31 = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_173 = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(mul_173, -1);  mul_173 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        sub_39 = torch.ops.aten.sub.Tensor(add_113, unsqueeze_249);  unsqueeze_249 = None
        mul_174 = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_251);  sub_39 = unsqueeze_251 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_175 = torch.ops.aten.mul.Tensor(mul_174, unsqueeze_253);  mul_174 = unsqueeze_253 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(arg23_1, -1);  arg23_1 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_115 = torch.ops.aten.add.Tensor(mul_175, unsqueeze_255);  mul_175 = unsqueeze_255 = None
        convolution_62 = torch.ops.aten.convolution.default(add_115, arg24_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_115 = arg24_1 = None
        mul_176 = torch.ops.aten.mul.Tensor(convolution_62, 0.5)
        mul_177 = torch.ops.aten.mul.Tensor(convolution_62, 0.7071067811865476);  convolution_62 = None
        erf_24 = torch.ops.aten.erf.default(mul_177);  mul_177 = None
        add_116 = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_178 = torch.ops.aten.mul.Tensor(mul_176, add_116);  mul_176 = add_116 = None
        convolution_63 = torch.ops.aten.convolution.default(mul_178, arg25_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_178 = arg25_1 = None
        mul_179 = torch.ops.aten.mul.Tensor(convolution_63, 0.5)
        mul_180 = torch.ops.aten.mul.Tensor(convolution_63, 0.7071067811865476);  convolution_63 = None
        erf_25 = torch.ops.aten.erf.default(mul_180);  mul_180 = None
        add_117 = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_179, add_117);  mul_179 = add_117 = None
        convolution_64 = torch.ops.aten.convolution.default(mul_181, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_181 = arg26_1 = None
        add_118 = torch.ops.aten.add.Tensor(add_113, convolution_64);  add_113 = convolution_64 = None
        add_119 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_32 = torch.ops.aten.sqrt.default(add_119);  add_119 = None
        reciprocal_32 = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_182 = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(mul_182, -1);  mul_182 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        sub_40 = torch.ops.aten.sub.Tensor(add_118, unsqueeze_257);  unsqueeze_257 = None
        mul_183 = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_259);  sub_40 = unsqueeze_259 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_184 = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_261);  mul_183 = unsqueeze_261 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_120 = torch.ops.aten.add.Tensor(mul_184, unsqueeze_263);  mul_184 = unsqueeze_263 = None
        convolution_65 = torch.ops.aten.convolution.default(add_120, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_120 = arg31_1 = None
        mul_185 = torch.ops.aten.mul.Tensor(convolution_65, 0.5)
        mul_186 = torch.ops.aten.mul.Tensor(convolution_65, 0.7071067811865476);  convolution_65 = None
        erf_26 = torch.ops.aten.erf.default(mul_186);  mul_186 = None
        add_121 = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_187 = torch.ops.aten.mul.Tensor(mul_185, add_121);  mul_185 = add_121 = None
        convolution_66 = torch.ops.aten.convolution.default(mul_187, arg32_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_187 = arg32_1 = None
        mul_188 = torch.ops.aten.mul.Tensor(convolution_66, 0.5)
        mul_189 = torch.ops.aten.mul.Tensor(convolution_66, 0.7071067811865476);  convolution_66 = None
        erf_27 = torch.ops.aten.erf.default(mul_189);  mul_189 = None
        add_122 = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_190 = torch.ops.aten.mul.Tensor(mul_188, add_122);  mul_188 = add_122 = None
        convolution_67 = torch.ops.aten.convolution.default(mul_190, arg33_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_190 = arg33_1 = None
        add_123 = torch.ops.aten.add.Tensor(add_118, convolution_67);  add_118 = convolution_67 = None
        add_124 = torch.ops.aten.add.Tensor(arg35_1, 1e-05);  arg35_1 = None
        sqrt_33 = torch.ops.aten.sqrt.default(add_124);  add_124 = None
        reciprocal_33 = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_191 = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(mul_191, -1);  mul_191 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        sub_41 = torch.ops.aten.sub.Tensor(add_123, unsqueeze_265);  unsqueeze_265 = None
        mul_192 = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_267);  sub_41 = unsqueeze_267 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_269);  mul_192 = unsqueeze_269 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_125 = torch.ops.aten.add.Tensor(mul_193, unsqueeze_271);  mul_193 = unsqueeze_271 = None
        convolution_68 = torch.ops.aten.convolution.default(add_125, arg38_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_125 = arg38_1 = None
        mul_194 = torch.ops.aten.mul.Tensor(convolution_68, 0.5)
        mul_195 = torch.ops.aten.mul.Tensor(convolution_68, 0.7071067811865476);  convolution_68 = None
        erf_28 = torch.ops.aten.erf.default(mul_195);  mul_195 = None
        add_126 = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_196 = torch.ops.aten.mul.Tensor(mul_194, add_126);  mul_194 = add_126 = None
        convolution_69 = torch.ops.aten.convolution.default(mul_196, arg39_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_196 = arg39_1 = None
        mul_197 = torch.ops.aten.mul.Tensor(convolution_69, 0.5)
        mul_198 = torch.ops.aten.mul.Tensor(convolution_69, 0.7071067811865476);  convolution_69 = None
        erf_29 = torch.ops.aten.erf.default(mul_198);  mul_198 = None
        add_127 = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_199 = torch.ops.aten.mul.Tensor(mul_197, add_127);  mul_197 = add_127 = None
        convolution_70 = torch.ops.aten.convolution.default(mul_199, arg40_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_199 = arg40_1 = None
        add_128 = torch.ops.aten.add.Tensor(add_123, convolution_70);  add_123 = convolution_70 = None
        add_129 = torch.ops.aten.add.Tensor(arg42_1, 1e-05);  arg42_1 = None
        sqrt_34 = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_34 = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_200 = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_272 = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(mul_200, -1);  mul_200 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        sub_42 = torch.ops.aten.sub.Tensor(add_128, unsqueeze_273);  unsqueeze_273 = None
        mul_201 = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_275);  sub_42 = unsqueeze_275 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_202 = torch.ops.aten.mul.Tensor(mul_201, unsqueeze_277);  mul_201 = unsqueeze_277 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_130 = torch.ops.aten.add.Tensor(mul_202, unsqueeze_279);  mul_202 = unsqueeze_279 = None
        convolution_71 = torch.ops.aten.convolution.default(add_130, arg45_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_130 = arg45_1 = None
        mul_203 = torch.ops.aten.mul.Tensor(convolution_71, 0.5)
        mul_204 = torch.ops.aten.mul.Tensor(convolution_71, 0.7071067811865476);  convolution_71 = None
        erf_30 = torch.ops.aten.erf.default(mul_204);  mul_204 = None
        add_131 = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_203, add_131);  mul_203 = add_131 = None
        convolution_72 = torch.ops.aten.convolution.default(mul_205, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_205 = arg46_1 = None
        mul_206 = torch.ops.aten.mul.Tensor(convolution_72, 0.5)
        mul_207 = torch.ops.aten.mul.Tensor(convolution_72, 0.7071067811865476);  convolution_72 = None
        erf_31 = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_132 = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_208 = torch.ops.aten.mul.Tensor(mul_206, add_132);  mul_206 = add_132 = None
        convolution_73 = torch.ops.aten.convolution.default(mul_208, arg47_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_208 = arg47_1 = None
        add_133 = torch.ops.aten.add.Tensor(add_128, convolution_73);  add_128 = convolution_73 = None
        add_134 = torch.ops.aten.add.Tensor(arg49_1, 1e-05);  arg49_1 = None
        sqrt_35 = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_35 = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_209 = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(arg48_1, -1);  arg48_1 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(mul_209, -1);  mul_209 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        sub_43 = torch.ops.aten.sub.Tensor(add_133, unsqueeze_281);  unsqueeze_281 = None
        mul_210 = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_283);  sub_43 = unsqueeze_283 = None
        unsqueeze_284 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_211 = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_285);  mul_210 = unsqueeze_285 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_135 = torch.ops.aten.add.Tensor(mul_211, unsqueeze_287);  mul_211 = unsqueeze_287 = None
        convolution_74 = torch.ops.aten.convolution.default(add_135, arg52_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_135 = arg52_1 = None
        mul_212 = torch.ops.aten.mul.Tensor(convolution_74, 0.5)
        mul_213 = torch.ops.aten.mul.Tensor(convolution_74, 0.7071067811865476);  convolution_74 = None
        erf_32 = torch.ops.aten.erf.default(mul_213);  mul_213 = None
        add_136 = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_212, add_136);  mul_212 = add_136 = None
        convolution_75 = torch.ops.aten.convolution.default(mul_214, arg53_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_214 = arg53_1 = None
        mul_215 = torch.ops.aten.mul.Tensor(convolution_75, 0.5)
        mul_216 = torch.ops.aten.mul.Tensor(convolution_75, 0.7071067811865476);  convolution_75 = None
        erf_33 = torch.ops.aten.erf.default(mul_216);  mul_216 = None
        add_137 = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_217 = torch.ops.aten.mul.Tensor(mul_215, add_137);  mul_215 = add_137 = None
        convolution_76 = torch.ops.aten.convolution.default(mul_217, arg54_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_217 = arg54_1 = None
        add_138 = torch.ops.aten.add.Tensor(add_133, convolution_76);  add_133 = convolution_76 = None
        add_139 = torch.ops.aten.add.Tensor(arg56_1, 1e-05);  arg56_1 = None
        sqrt_36 = torch.ops.aten.sqrt.default(add_139);  add_139 = None
        reciprocal_36 = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_218 = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(mul_218, -1);  mul_218 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        sub_44 = torch.ops.aten.sub.Tensor(add_138, unsqueeze_289);  unsqueeze_289 = None
        mul_219 = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_291);  sub_44 = unsqueeze_291 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_220 = torch.ops.aten.mul.Tensor(mul_219, unsqueeze_293);  mul_219 = unsqueeze_293 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(arg58_1, -1);  arg58_1 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_140 = torch.ops.aten.add.Tensor(mul_220, unsqueeze_295);  mul_220 = unsqueeze_295 = None
        convolution_77 = torch.ops.aten.convolution.default(add_140, arg59_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_140 = arg59_1 = None
        mul_221 = torch.ops.aten.mul.Tensor(convolution_77, 0.5)
        mul_222 = torch.ops.aten.mul.Tensor(convolution_77, 0.7071067811865476);  convolution_77 = None
        erf_34 = torch.ops.aten.erf.default(mul_222);  mul_222 = None
        add_141 = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_223 = torch.ops.aten.mul.Tensor(mul_221, add_141);  mul_221 = add_141 = None
        convolution_78 = torch.ops.aten.convolution.default(mul_223, arg60_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_223 = arg60_1 = None
        mul_224 = torch.ops.aten.mul.Tensor(convolution_78, 0.5)
        mul_225 = torch.ops.aten.mul.Tensor(convolution_78, 0.7071067811865476);  convolution_78 = None
        erf_35 = torch.ops.aten.erf.default(mul_225);  mul_225 = None
        add_142 = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_226 = torch.ops.aten.mul.Tensor(mul_224, add_142);  mul_224 = add_142 = None
        convolution_79 = torch.ops.aten.convolution.default(mul_226, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_226 = arg61_1 = None
        add_143 = torch.ops.aten.add.Tensor(add_138, convolution_79);  add_138 = convolution_79 = None
        convolution_80 = torch.ops.aten.convolution.default(add_143, arg62_1, arg63_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  add_143 = arg62_1 = arg63_1 = None
        add_144 = torch.ops.aten.add.Tensor(arg65_1, 1e-05);  arg65_1 = None
        sqrt_37 = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_37 = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_227 = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_296 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(mul_227, -1);  mul_227 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_297);  convolution_80 = unsqueeze_297 = None
        mul_228 = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_299);  sub_45 = unsqueeze_299 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(arg66_1, -1);  arg66_1 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_229 = torch.ops.aten.mul.Tensor(mul_228, unsqueeze_301);  mul_228 = unsqueeze_301 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_145 = torch.ops.aten.add.Tensor(mul_229, unsqueeze_303);  mul_229 = unsqueeze_303 = None
        add_146 = torch.ops.aten.add.Tensor(add_145, arg68_1);  add_145 = arg68_1 = None
        add_147 = torch.ops.aten.add.Tensor(arg70_1, 1e-05);  arg70_1 = None
        sqrt_38 = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_230 = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(mul_230, -1);  mul_230 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_46 = torch.ops.aten.sub.Tensor(add_146, unsqueeze_305);  unsqueeze_305 = None
        mul_231 = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_307);  sub_46 = unsqueeze_307 = None
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(arg71_1, -1);  arg71_1 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_232 = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_309);  mul_231 = unsqueeze_309 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_148 = torch.ops.aten.add.Tensor(mul_232, unsqueeze_311);  mul_232 = unsqueeze_311 = None
        convolution_81 = torch.ops.aten.convolution.default(add_148, arg73_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_148 = arg73_1 = None
        view_65 = torch.ops.aten.view.default(convolution_81, [8, 3, 6, 64, -1]);  convolution_81 = None
        permute_25 = torch.ops.aten.permute.default(view_65, [1, 0, 2, 4, 3]);  view_65 = None
        unbind_8 = torch.ops.aten.unbind.int(permute_25);  permute_25 = None
        getitem_24 = unbind_8[0]
        getitem_25 = unbind_8[1]
        getitem_26 = unbind_8[2];  unbind_8 = None
        permute_26 = torch.ops.aten.permute.default(getitem_25, [0, 1, 3, 2]);  getitem_25 = None
        expand_32 = torch.ops.aten.expand.default(getitem_24, [8, 6, 196, 64]);  getitem_24 = None
        clone_98 = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_66 = torch.ops.aten.view.default(clone_98, [48, 196, 64]);  clone_98 = None
        expand_33 = torch.ops.aten.expand.default(permute_26, [8, 6, 64, 196]);  permute_26 = None
        clone_99 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_67 = torch.ops.aten.view.default(clone_99, [48, 64, 196]);  clone_99 = None
        bmm_16 = torch.ops.aten.bmm.default(view_66, view_67);  view_66 = view_67 = None
        view_68 = torch.ops.aten.view.default(bmm_16, [8, 6, 196, 196]);  bmm_16 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(view_68, 1);  view_68 = None
        amax_default_7 = torch.ops.aten.amax.default(mul_tensor_14, [-1], True)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(mul_tensor_14, amax_default_7);  mul_tensor_14 = amax_default_7 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_7, 0.125);  sub_tensor_7 = None
        exp_8 = torch.ops.aten.exp.default(mul_tensor_15);  mul_tensor_15 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8 = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        expand_34 = torch.ops.aten.expand.default(div_8, [8, 6, 196, 196]);  div_8 = None
        view_69 = torch.ops.aten.view.default(expand_34, [48, 196, 196]);  expand_34 = None
        expand_35 = torch.ops.aten.expand.default(getitem_26, [8, 6, 196, 64]);  getitem_26 = None
        clone_101 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_70 = torch.ops.aten.view.default(clone_101, [48, 196, 64]);  clone_101 = None
        bmm_17 = torch.ops.aten.bmm.default(view_69, view_70);  view_69 = view_70 = None
        view_71 = torch.ops.aten.view.default(bmm_17, [8, 6, 196, 64]);  bmm_17 = None
        permute_27 = torch.ops.aten.permute.default(view_71, [0, 1, 3, 2]);  view_71 = None
        clone_102 = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_72 = torch.ops.aten.view.default(clone_102, [8, 384, 14, 14]);  clone_102 = None
        convolution_82 = torch.ops.aten.convolution.default(view_72, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_72 = arg74_1 = None
        add_149 = torch.ops.aten.add.Tensor(add_146, convolution_82);  add_146 = convolution_82 = None
        add_150 = torch.ops.aten.add.Tensor(arg76_1, 1e-05);  arg76_1 = None
        sqrt_39 = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_39 = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_234 = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_48 = torch.ops.aten.sub.Tensor(add_149, unsqueeze_313);  unsqueeze_313 = None
        mul_235 = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_315);  sub_48 = unsqueeze_315 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_317);  mul_235 = unsqueeze_317 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_151 = torch.ops.aten.add.Tensor(mul_236, unsqueeze_319);  mul_236 = unsqueeze_319 = None
        convolution_83 = torch.ops.aten.convolution.default(add_151, arg79_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_151 = arg79_1 = None
        mul_237 = torch.ops.aten.mul.Tensor(convolution_83, 0.5)
        mul_238 = torch.ops.aten.mul.Tensor(convolution_83, 0.7071067811865476);  convolution_83 = None
        erf_36 = torch.ops.aten.erf.default(mul_238);  mul_238 = None
        add_152 = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_237, add_152);  mul_237 = add_152 = None
        convolution_84 = torch.ops.aten.convolution.default(mul_239, arg80_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_239 = arg80_1 = None
        add_153 = torch.ops.aten.add.Tensor(add_149, convolution_84);  add_149 = convolution_84 = None
        add_154 = torch.ops.aten.add.Tensor(arg82_1, 1e-05);  arg82_1 = None
        sqrt_40 = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_40 = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_240 = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(arg81_1, -1);  arg81_1 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_49 = torch.ops.aten.sub.Tensor(add_153, unsqueeze_321);  unsqueeze_321 = None
        mul_241 = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_323);  sub_49 = unsqueeze_323 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_325);  mul_241 = unsqueeze_325 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_155 = torch.ops.aten.add.Tensor(mul_242, unsqueeze_327);  mul_242 = unsqueeze_327 = None
        convolution_85 = torch.ops.aten.convolution.default(add_155, arg85_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_155 = arg85_1 = None
        view_73 = torch.ops.aten.view.default(convolution_85, [8, 3, 6, 64, -1]);  convolution_85 = None
        permute_28 = torch.ops.aten.permute.default(view_73, [1, 0, 2, 4, 3]);  view_73 = None
        unbind_9 = torch.ops.aten.unbind.int(permute_28);  permute_28 = None
        getitem_27 = unbind_9[0]
        getitem_28 = unbind_9[1]
        getitem_29 = unbind_9[2];  unbind_9 = None
        permute_29 = torch.ops.aten.permute.default(getitem_28, [0, 1, 3, 2]);  getitem_28 = None
        expand_36 = torch.ops.aten.expand.default(getitem_27, [8, 6, 196, 64]);  getitem_27 = None
        clone_106 = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_74 = torch.ops.aten.view.default(clone_106, [48, 196, 64]);  clone_106 = None
        expand_37 = torch.ops.aten.expand.default(permute_29, [8, 6, 64, 196]);  permute_29 = None
        clone_107 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_75 = torch.ops.aten.view.default(clone_107, [48, 64, 196]);  clone_107 = None
        bmm_18 = torch.ops.aten.bmm.default(view_74, view_75);  view_74 = view_75 = None
        view_76 = torch.ops.aten.view.default(bmm_18, [8, 6, 196, 196]);  bmm_18 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(view_76, 1);  view_76 = None
        amax_default_6 = torch.ops.aten.amax.default(mul_tensor_12, [-1], True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(mul_tensor_12, amax_default_6);  mul_tensor_12 = amax_default_6 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.125);  sub_tensor_6 = None
        exp_9 = torch.ops.aten.exp.default(mul_tensor_13);  mul_tensor_13 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9 = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        expand_38 = torch.ops.aten.expand.default(div_9, [8, 6, 196, 196]);  div_9 = None
        view_77 = torch.ops.aten.view.default(expand_38, [48, 196, 196]);  expand_38 = None
        expand_39 = torch.ops.aten.expand.default(getitem_29, [8, 6, 196, 64]);  getitem_29 = None
        clone_109 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_78 = torch.ops.aten.view.default(clone_109, [48, 196, 64]);  clone_109 = None
        bmm_19 = torch.ops.aten.bmm.default(view_77, view_78);  view_77 = view_78 = None
        view_79 = torch.ops.aten.view.default(bmm_19, [8, 6, 196, 64]);  bmm_19 = None
        permute_30 = torch.ops.aten.permute.default(view_79, [0, 1, 3, 2]);  view_79 = None
        clone_110 = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_80 = torch.ops.aten.view.default(clone_110, [8, 384, 14, 14]);  clone_110 = None
        convolution_86 = torch.ops.aten.convolution.default(view_80, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_80 = arg86_1 = None
        add_156 = torch.ops.aten.add.Tensor(add_153, convolution_86);  add_153 = convolution_86 = None
        add_157 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_244 = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_244, -1);  mul_244 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_51 = torch.ops.aten.sub.Tensor(add_156, unsqueeze_329);  unsqueeze_329 = None
        mul_245 = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_331);  sub_51 = unsqueeze_331 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_246 = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_333);  mul_245 = unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_158 = torch.ops.aten.add.Tensor(mul_246, unsqueeze_335);  mul_246 = unsqueeze_335 = None
        convolution_87 = torch.ops.aten.convolution.default(add_158, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_158 = arg91_1 = None
        mul_247 = torch.ops.aten.mul.Tensor(convolution_87, 0.5)
        mul_248 = torch.ops.aten.mul.Tensor(convolution_87, 0.7071067811865476);  convolution_87 = None
        erf_37 = torch.ops.aten.erf.default(mul_248);  mul_248 = None
        add_159 = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_247, add_159);  mul_247 = add_159 = None
        convolution_88 = torch.ops.aten.convolution.default(mul_249, arg92_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_249 = arg92_1 = None
        add_160 = torch.ops.aten.add.Tensor(add_156, convolution_88);  add_156 = convolution_88 = None
        add_161 = torch.ops.aten.add.Tensor(arg94_1, 1e-05);  arg94_1 = None
        sqrt_42 = torch.ops.aten.sqrt.default(add_161);  add_161 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_250 = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_250, -1);  mul_250 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_52 = torch.ops.aten.sub.Tensor(add_160, unsqueeze_337);  unsqueeze_337 = None
        mul_251 = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_339);  sub_52 = unsqueeze_339 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_252 = torch.ops.aten.mul.Tensor(mul_251, unsqueeze_341);  mul_251 = unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_162 = torch.ops.aten.add.Tensor(mul_252, unsqueeze_343);  mul_252 = unsqueeze_343 = None
        convolution_89 = torch.ops.aten.convolution.default(add_162, arg97_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_162 = arg97_1 = None
        view_81 = torch.ops.aten.view.default(convolution_89, [8, 3, 6, 64, -1]);  convolution_89 = None
        permute_31 = torch.ops.aten.permute.default(view_81, [1, 0, 2, 4, 3]);  view_81 = None
        unbind_10 = torch.ops.aten.unbind.int(permute_31);  permute_31 = None
        getitem_30 = unbind_10[0]
        getitem_31 = unbind_10[1]
        getitem_32 = unbind_10[2];  unbind_10 = None
        permute_32 = torch.ops.aten.permute.default(getitem_31, [0, 1, 3, 2]);  getitem_31 = None
        expand_40 = torch.ops.aten.expand.default(getitem_30, [8, 6, 196, 64]);  getitem_30 = None
        clone_114 = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_82 = torch.ops.aten.view.default(clone_114, [48, 196, 64]);  clone_114 = None
        expand_41 = torch.ops.aten.expand.default(permute_32, [8, 6, 64, 196]);  permute_32 = None
        clone_115 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_83 = torch.ops.aten.view.default(clone_115, [48, 64, 196]);  clone_115 = None
        bmm_20 = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
        view_84 = torch.ops.aten.view.default(bmm_20, [8, 6, 196, 196]);  bmm_20 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_84, 1);  view_84 = None
        amax_default_5 = torch.ops.aten.amax.default(mul_tensor_10, [-1], True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(mul_tensor_10, amax_default_5);  mul_tensor_10 = amax_default_5 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_5, 0.125);  sub_tensor_5 = None
        exp_10 = torch.ops.aten.exp.default(mul_tensor_11);  mul_tensor_11 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10 = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        expand_42 = torch.ops.aten.expand.default(div_10, [8, 6, 196, 196]);  div_10 = None
        view_85 = torch.ops.aten.view.default(expand_42, [48, 196, 196]);  expand_42 = None
        expand_43 = torch.ops.aten.expand.default(getitem_32, [8, 6, 196, 64]);  getitem_32 = None
        clone_117 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_86 = torch.ops.aten.view.default(clone_117, [48, 196, 64]);  clone_117 = None
        bmm_21 = torch.ops.aten.bmm.default(view_85, view_86);  view_85 = view_86 = None
        view_87 = torch.ops.aten.view.default(bmm_21, [8, 6, 196, 64]);  bmm_21 = None
        permute_33 = torch.ops.aten.permute.default(view_87, [0, 1, 3, 2]);  view_87 = None
        clone_118 = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_88 = torch.ops.aten.view.default(clone_118, [8, 384, 14, 14]);  clone_118 = None
        convolution_90 = torch.ops.aten.convolution.default(view_88, arg98_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_88 = arg98_1 = None
        add_163 = torch.ops.aten.add.Tensor(add_160, convolution_90);  add_160 = convolution_90 = None
        add_164 = torch.ops.aten.add.Tensor(arg100_1, 1e-05);  arg100_1 = None
        sqrt_43 = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_254 = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(mul_254, -1);  mul_254 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_54 = torch.ops.aten.sub.Tensor(add_163, unsqueeze_345);  unsqueeze_345 = None
        mul_255 = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_347);  sub_54 = unsqueeze_347 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(arg101_1, -1);  arg101_1 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_256 = torch.ops.aten.mul.Tensor(mul_255, unsqueeze_349);  mul_255 = unsqueeze_349 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_165 = torch.ops.aten.add.Tensor(mul_256, unsqueeze_351);  mul_256 = unsqueeze_351 = None
        convolution_91 = torch.ops.aten.convolution.default(add_165, arg103_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_165 = arg103_1 = None
        mul_257 = torch.ops.aten.mul.Tensor(convolution_91, 0.5)
        mul_258 = torch.ops.aten.mul.Tensor(convolution_91, 0.7071067811865476);  convolution_91 = None
        erf_38 = torch.ops.aten.erf.default(mul_258);  mul_258 = None
        add_166 = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_259 = torch.ops.aten.mul.Tensor(mul_257, add_166);  mul_257 = add_166 = None
        convolution_92 = torch.ops.aten.convolution.default(mul_259, arg104_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_259 = arg104_1 = None
        add_167 = torch.ops.aten.add.Tensor(add_163, convolution_92);  add_163 = convolution_92 = None
        add_168 = torch.ops.aten.add.Tensor(arg106_1, 1e-05);  arg106_1 = None
        sqrt_44 = torch.ops.aten.sqrt.default(add_168);  add_168 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_260 = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_260, -1);  mul_260 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_55 = torch.ops.aten.sub.Tensor(add_167, unsqueeze_353);  unsqueeze_353 = None
        mul_261 = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_355);  sub_55 = unsqueeze_355 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_262 = torch.ops.aten.mul.Tensor(mul_261, unsqueeze_357);  mul_261 = unsqueeze_357 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(arg108_1, -1);  arg108_1 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_169 = torch.ops.aten.add.Tensor(mul_262, unsqueeze_359);  mul_262 = unsqueeze_359 = None
        convolution_93 = torch.ops.aten.convolution.default(add_169, arg109_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_169 = arg109_1 = None
        view_89 = torch.ops.aten.view.default(convolution_93, [8, 3, 6, 64, -1]);  convolution_93 = None
        permute_34 = torch.ops.aten.permute.default(view_89, [1, 0, 2, 4, 3]);  view_89 = None
        unbind_11 = torch.ops.aten.unbind.int(permute_34);  permute_34 = None
        getitem_33 = unbind_11[0]
        getitem_34 = unbind_11[1]
        getitem_35 = unbind_11[2];  unbind_11 = None
        permute_35 = torch.ops.aten.permute.default(getitem_34, [0, 1, 3, 2]);  getitem_34 = None
        expand_44 = torch.ops.aten.expand.default(getitem_33, [8, 6, 196, 64]);  getitem_33 = None
        clone_122 = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_90 = torch.ops.aten.view.default(clone_122, [48, 196, 64]);  clone_122 = None
        expand_45 = torch.ops.aten.expand.default(permute_35, [8, 6, 64, 196]);  permute_35 = None
        clone_123 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_91 = torch.ops.aten.view.default(clone_123, [48, 64, 196]);  clone_123 = None
        bmm_22 = torch.ops.aten.bmm.default(view_90, view_91);  view_90 = view_91 = None
        view_92 = torch.ops.aten.view.default(bmm_22, [8, 6, 196, 196]);  bmm_22 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_92, 1);  view_92 = None
        amax_default_4 = torch.ops.aten.amax.default(mul_tensor_8, [-1], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_8, amax_default_4);  mul_tensor_8 = amax_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.125);  sub_tensor_4 = None
        exp_11 = torch.ops.aten.exp.default(mul_tensor_9);  mul_tensor_9 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11 = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        expand_46 = torch.ops.aten.expand.default(div_11, [8, 6, 196, 196]);  div_11 = None
        view_93 = torch.ops.aten.view.default(expand_46, [48, 196, 196]);  expand_46 = None
        expand_47 = torch.ops.aten.expand.default(getitem_35, [8, 6, 196, 64]);  getitem_35 = None
        clone_125 = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_94 = torch.ops.aten.view.default(clone_125, [48, 196, 64]);  clone_125 = None
        bmm_23 = torch.ops.aten.bmm.default(view_93, view_94);  view_93 = view_94 = None
        view_95 = torch.ops.aten.view.default(bmm_23, [8, 6, 196, 64]);  bmm_23 = None
        permute_36 = torch.ops.aten.permute.default(view_95, [0, 1, 3, 2]);  view_95 = None
        clone_126 = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_96 = torch.ops.aten.view.default(clone_126, [8, 384, 14, 14]);  clone_126 = None
        convolution_94 = torch.ops.aten.convolution.default(view_96, arg110_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_96 = arg110_1 = None
        add_170 = torch.ops.aten.add.Tensor(add_167, convolution_94);  add_167 = convolution_94 = None
        add_171 = torch.ops.aten.add.Tensor(arg112_1, 1e-05);  arg112_1 = None
        sqrt_45 = torch.ops.aten.sqrt.default(add_171);  add_171 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_264 = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(arg111_1, -1);  arg111_1 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_57 = torch.ops.aten.sub.Tensor(add_170, unsqueeze_361);  unsqueeze_361 = None
        mul_265 = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_363);  sub_57 = unsqueeze_363 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(arg113_1, -1);  arg113_1 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_365);  mul_265 = unsqueeze_365 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_172 = torch.ops.aten.add.Tensor(mul_266, unsqueeze_367);  mul_266 = unsqueeze_367 = None
        convolution_95 = torch.ops.aten.convolution.default(add_172, arg115_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_172 = arg115_1 = None
        mul_267 = torch.ops.aten.mul.Tensor(convolution_95, 0.5)
        mul_268 = torch.ops.aten.mul.Tensor(convolution_95, 0.7071067811865476);  convolution_95 = None
        erf_39 = torch.ops.aten.erf.default(mul_268);  mul_268 = None
        add_173 = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_267, add_173);  mul_267 = add_173 = None
        convolution_96 = torch.ops.aten.convolution.default(mul_269, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_269 = arg116_1 = None
        add_174 = torch.ops.aten.add.Tensor(add_170, convolution_96);  add_170 = convolution_96 = None
        convolution_97 = torch.ops.aten.convolution.default(add_174, arg117_1, arg118_1, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  add_174 = arg117_1 = arg118_1 = None
        add_175 = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
        sqrt_46 = torch.ops.aten.sqrt.default(add_175);  add_175 = None
        reciprocal_46 = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_270 = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_58 = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_369);  convolution_97 = unsqueeze_369 = None
        mul_271 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_371);  sub_58 = unsqueeze_371 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_373);  mul_271 = unsqueeze_373 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_176 = torch.ops.aten.add.Tensor(mul_272, unsqueeze_375);  mul_272 = unsqueeze_375 = None
        add_177 = torch.ops.aten.add.Tensor(add_176, arg123_1);  add_176 = arg123_1 = None
        add_178 = torch.ops.aten.add.Tensor(arg125_1, 1e-05);  arg125_1 = None
        sqrt_47 = torch.ops.aten.sqrt.default(add_178);  add_178 = None
        reciprocal_47 = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_273 = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_59 = torch.ops.aten.sub.Tensor(add_177, unsqueeze_377);  unsqueeze_377 = None
        mul_274 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_379);  sub_59 = unsqueeze_379 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_381);  mul_274 = unsqueeze_381 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_179 = torch.ops.aten.add.Tensor(mul_275, unsqueeze_383);  mul_275 = unsqueeze_383 = None
        convolution_98 = torch.ops.aten.convolution.default(add_179, arg128_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_179 = arg128_1 = None
        view_97 = torch.ops.aten.view.default(convolution_98, [8, 3, 6, 128, -1]);  convolution_98 = None
        permute_37 = torch.ops.aten.permute.default(view_97, [1, 0, 2, 4, 3]);  view_97 = None
        unbind_12 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
        getitem_36 = unbind_12[0]
        getitem_37 = unbind_12[1]
        getitem_38 = unbind_12[2];  unbind_12 = None
        permute_38 = torch.ops.aten.permute.default(getitem_37, [0, 1, 3, 2]);  getitem_37 = None
        expand_48 = torch.ops.aten.expand.default(getitem_36, [8, 6, 49, 128]);  getitem_36 = None
        clone_131 = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_98 = torch.ops.aten.view.default(clone_131, [48, 49, 128]);  clone_131 = None
        expand_49 = torch.ops.aten.expand.default(permute_38, [8, 6, 128, 49]);  permute_38 = None
        clone_132 = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_99 = torch.ops.aten.view.default(clone_132, [48, 128, 49]);  clone_132 = None
        bmm_24 = torch.ops.aten.bmm.default(view_98, view_99);  view_98 = view_99 = None
        view_100 = torch.ops.aten.view.default(bmm_24, [8, 6, 49, 49]);  bmm_24 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(view_100, 1);  view_100 = None
        amax_default_3 = torch.ops.aten.amax.default(mul_tensor_6, [-1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, amax_default_3);  mul_tensor_6 = amax_default_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, 0.08838834764831845);  sub_tensor_3 = None
        exp_12 = torch.ops.aten.exp.default(mul_tensor_7);  mul_tensor_7 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12 = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        expand_50 = torch.ops.aten.expand.default(div_12, [8, 6, 49, 49]);  div_12 = None
        view_101 = torch.ops.aten.view.default(expand_50, [48, 49, 49]);  expand_50 = None
        expand_51 = torch.ops.aten.expand.default(getitem_38, [8, 6, 49, 128]);  getitem_38 = None
        clone_134 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_102 = torch.ops.aten.view.default(clone_134, [48, 49, 128]);  clone_134 = None
        bmm_25 = torch.ops.aten.bmm.default(view_101, view_102);  view_101 = view_102 = None
        view_103 = torch.ops.aten.view.default(bmm_25, [8, 6, 49, 128]);  bmm_25 = None
        permute_39 = torch.ops.aten.permute.default(view_103, [0, 1, 3, 2]);  view_103 = None
        clone_135 = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_104 = torch.ops.aten.view.default(clone_135, [8, 768, 7, 7]);  clone_135 = None
        convolution_99 = torch.ops.aten.convolution.default(view_104, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_104 = arg129_1 = None
        add_180 = torch.ops.aten.add.Tensor(add_177, convolution_99);  add_177 = convolution_99 = None
        add_181 = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_48 = torch.ops.aten.sqrt.default(add_181);  add_181 = None
        reciprocal_48 = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_277 = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(mul_277, -1);  mul_277 = None
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_61 = torch.ops.aten.sub.Tensor(add_180, unsqueeze_385);  unsqueeze_385 = None
        mul_278 = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_387);  sub_61 = unsqueeze_387 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_279 = torch.ops.aten.mul.Tensor(mul_278, unsqueeze_389);  mul_278 = unsqueeze_389 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_182 = torch.ops.aten.add.Tensor(mul_279, unsqueeze_391);  mul_279 = unsqueeze_391 = None
        convolution_100 = torch.ops.aten.convolution.default(add_182, arg134_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_182 = arg134_1 = None
        mul_280 = torch.ops.aten.mul.Tensor(convolution_100, 0.5)
        mul_281 = torch.ops.aten.mul.Tensor(convolution_100, 0.7071067811865476);  convolution_100 = None
        erf_40 = torch.ops.aten.erf.default(mul_281);  mul_281 = None
        add_183 = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
        mul_282 = torch.ops.aten.mul.Tensor(mul_280, add_183);  mul_280 = add_183 = None
        convolution_101 = torch.ops.aten.convolution.default(mul_282, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_282 = arg135_1 = None
        add_184 = torch.ops.aten.add.Tensor(add_180, convolution_101);  add_180 = convolution_101 = None
        add_185 = torch.ops.aten.add.Tensor(arg137_1, 1e-05);  arg137_1 = None
        sqrt_49 = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_49 = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_283 = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(mul_283, -1);  mul_283 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_62 = torch.ops.aten.sub.Tensor(add_184, unsqueeze_393);  unsqueeze_393 = None
        mul_284 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_395);  sub_62 = unsqueeze_395 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_285 = torch.ops.aten.mul.Tensor(mul_284, unsqueeze_397);  mul_284 = unsqueeze_397 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_186 = torch.ops.aten.add.Tensor(mul_285, unsqueeze_399);  mul_285 = unsqueeze_399 = None
        convolution_102 = torch.ops.aten.convolution.default(add_186, arg140_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_186 = arg140_1 = None
        view_105 = torch.ops.aten.view.default(convolution_102, [8, 3, 6, 128, -1]);  convolution_102 = None
        permute_40 = torch.ops.aten.permute.default(view_105, [1, 0, 2, 4, 3]);  view_105 = None
        unbind_13 = torch.ops.aten.unbind.int(permute_40);  permute_40 = None
        getitem_39 = unbind_13[0]
        getitem_40 = unbind_13[1]
        getitem_41 = unbind_13[2];  unbind_13 = None
        permute_41 = torch.ops.aten.permute.default(getitem_40, [0, 1, 3, 2]);  getitem_40 = None
        expand_52 = torch.ops.aten.expand.default(getitem_39, [8, 6, 49, 128]);  getitem_39 = None
        clone_139 = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_106 = torch.ops.aten.view.default(clone_139, [48, 49, 128]);  clone_139 = None
        expand_53 = torch.ops.aten.expand.default(permute_41, [8, 6, 128, 49]);  permute_41 = None
        clone_140 = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_107 = torch.ops.aten.view.default(clone_140, [48, 128, 49]);  clone_140 = None
        bmm_26 = torch.ops.aten.bmm.default(view_106, view_107);  view_106 = view_107 = None
        view_108 = torch.ops.aten.view.default(bmm_26, [8, 6, 49, 49]);  bmm_26 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_108, 1);  view_108 = None
        amax_default_2 = torch.ops.aten.amax.default(mul_tensor_4, [-1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_4, amax_default_2);  mul_tensor_4 = amax_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.08838834764831845);  sub_tensor_2 = None
        exp_13 = torch.ops.aten.exp.default(mul_tensor_5);  mul_tensor_5 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13 = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        expand_54 = torch.ops.aten.expand.default(div_13, [8, 6, 49, 49]);  div_13 = None
        view_109 = torch.ops.aten.view.default(expand_54, [48, 49, 49]);  expand_54 = None
        expand_55 = torch.ops.aten.expand.default(getitem_41, [8, 6, 49, 128]);  getitem_41 = None
        clone_142 = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_110 = torch.ops.aten.view.default(clone_142, [48, 49, 128]);  clone_142 = None
        bmm_27 = torch.ops.aten.bmm.default(view_109, view_110);  view_109 = view_110 = None
        view_111 = torch.ops.aten.view.default(bmm_27, [8, 6, 49, 128]);  bmm_27 = None
        permute_42 = torch.ops.aten.permute.default(view_111, [0, 1, 3, 2]);  view_111 = None
        clone_143 = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_112 = torch.ops.aten.view.default(clone_143, [8, 768, 7, 7]);  clone_143 = None
        convolution_103 = torch.ops.aten.convolution.default(view_112, arg141_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_112 = arg141_1 = None
        add_187 = torch.ops.aten.add.Tensor(add_184, convolution_103);  add_184 = convolution_103 = None
        add_188 = torch.ops.aten.add.Tensor(arg143_1, 1e-05);  arg143_1 = None
        sqrt_50 = torch.ops.aten.sqrt.default(add_188);  add_188 = None
        reciprocal_50 = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_287 = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_287, -1);  mul_287 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_64 = torch.ops.aten.sub.Tensor(add_187, unsqueeze_401);  unsqueeze_401 = None
        mul_288 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_403);  sub_64 = unsqueeze_403 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_289 = torch.ops.aten.mul.Tensor(mul_288, unsqueeze_405);  mul_288 = unsqueeze_405 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_189 = torch.ops.aten.add.Tensor(mul_289, unsqueeze_407);  mul_289 = unsqueeze_407 = None
        convolution_104 = torch.ops.aten.convolution.default(add_189, arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_189 = arg146_1 = None
        mul_290 = torch.ops.aten.mul.Tensor(convolution_104, 0.5)
        mul_291 = torch.ops.aten.mul.Tensor(convolution_104, 0.7071067811865476);  convolution_104 = None
        erf_41 = torch.ops.aten.erf.default(mul_291);  mul_291 = None
        add_190 = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
        mul_292 = torch.ops.aten.mul.Tensor(mul_290, add_190);  mul_290 = add_190 = None
        convolution_105 = torch.ops.aten.convolution.default(mul_292, arg147_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_292 = arg147_1 = None
        add_191 = torch.ops.aten.add.Tensor(add_187, convolution_105);  add_187 = convolution_105 = None
        add_192 = torch.ops.aten.add.Tensor(arg149_1, 1e-05);  arg149_1 = None
        sqrt_51 = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_51 = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_293 = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(mul_293, -1);  mul_293 = None
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_65 = torch.ops.aten.sub.Tensor(add_191, unsqueeze_409);  unsqueeze_409 = None
        mul_294 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_411);  sub_65 = unsqueeze_411 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_295 = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_413);  mul_294 = unsqueeze_413 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_193 = torch.ops.aten.add.Tensor(mul_295, unsqueeze_415);  mul_295 = unsqueeze_415 = None
        convolution_106 = torch.ops.aten.convolution.default(add_193, arg152_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_193 = arg152_1 = None
        view_113 = torch.ops.aten.view.default(convolution_106, [8, 3, 6, 128, -1]);  convolution_106 = None
        permute_43 = torch.ops.aten.permute.default(view_113, [1, 0, 2, 4, 3]);  view_113 = None
        unbind_14 = torch.ops.aten.unbind.int(permute_43);  permute_43 = None
        getitem_42 = unbind_14[0]
        getitem_43 = unbind_14[1]
        getitem_44 = unbind_14[2];  unbind_14 = None
        permute_44 = torch.ops.aten.permute.default(getitem_43, [0, 1, 3, 2]);  getitem_43 = None
        expand_56 = torch.ops.aten.expand.default(getitem_42, [8, 6, 49, 128]);  getitem_42 = None
        clone_147 = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_114 = torch.ops.aten.view.default(clone_147, [48, 49, 128]);  clone_147 = None
        expand_57 = torch.ops.aten.expand.default(permute_44, [8, 6, 128, 49]);  permute_44 = None
        clone_148 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_115 = torch.ops.aten.view.default(clone_148, [48, 128, 49]);  clone_148 = None
        bmm_28 = torch.ops.aten.bmm.default(view_114, view_115);  view_114 = view_115 = None
        view_116 = torch.ops.aten.view.default(bmm_28, [8, 6, 49, 49]);  bmm_28 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_116, 1);  view_116 = None
        amax_default_1 = torch.ops.aten.amax.default(mul_tensor_2, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, amax_default_1);  mul_tensor_2 = amax_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, 0.08838834764831845);  sub_tensor_1 = None
        exp_14 = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14 = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        expand_58 = torch.ops.aten.expand.default(div_14, [8, 6, 49, 49]);  div_14 = None
        view_117 = torch.ops.aten.view.default(expand_58, [48, 49, 49]);  expand_58 = None
        expand_59 = torch.ops.aten.expand.default(getitem_44, [8, 6, 49, 128]);  getitem_44 = None
        clone_150 = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_118 = torch.ops.aten.view.default(clone_150, [48, 49, 128]);  clone_150 = None
        bmm_29 = torch.ops.aten.bmm.default(view_117, view_118);  view_117 = view_118 = None
        view_119 = torch.ops.aten.view.default(bmm_29, [8, 6, 49, 128]);  bmm_29 = None
        permute_45 = torch.ops.aten.permute.default(view_119, [0, 1, 3, 2]);  view_119 = None
        clone_151 = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_120 = torch.ops.aten.view.default(clone_151, [8, 768, 7, 7]);  clone_151 = None
        convolution_107 = torch.ops.aten.convolution.default(view_120, arg153_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_120 = arg153_1 = None
        add_194 = torch.ops.aten.add.Tensor(add_191, convolution_107);  add_191 = convolution_107 = None
        add_195 = torch.ops.aten.add.Tensor(arg155_1, 1e-05);  arg155_1 = None
        sqrt_52 = torch.ops.aten.sqrt.default(add_195);  add_195 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_297 = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_67 = torch.ops.aten.sub.Tensor(add_194, unsqueeze_417);  unsqueeze_417 = None
        mul_298 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_419);  sub_67 = unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_421);  mul_298 = unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(arg157_1, -1);  arg157_1 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_196 = torch.ops.aten.add.Tensor(mul_299, unsqueeze_423);  mul_299 = unsqueeze_423 = None
        convolution_108 = torch.ops.aten.convolution.default(add_196, arg158_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_196 = arg158_1 = None
        mul_300 = torch.ops.aten.mul.Tensor(convolution_108, 0.5)
        mul_301 = torch.ops.aten.mul.Tensor(convolution_108, 0.7071067811865476);  convolution_108 = None
        erf_42 = torch.ops.aten.erf.default(mul_301);  mul_301 = None
        add_197 = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_300, add_197);  mul_300 = add_197 = None
        convolution_109 = torch.ops.aten.convolution.default(mul_302, arg159_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_302 = arg159_1 = None
        add_198 = torch.ops.aten.add.Tensor(add_194, convolution_109);  add_194 = convolution_109 = None
        add_199 = torch.ops.aten.add.Tensor(arg161_1, 1e-05);  arg161_1 = None
        sqrt_53 = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_53 = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_303 = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_68 = torch.ops.aten.sub.Tensor(add_198, unsqueeze_425);  unsqueeze_425 = None
        mul_304 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_427);  sub_68 = unsqueeze_427 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(arg162_1, -1);  arg162_1 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_429);  mul_304 = unsqueeze_429 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_200 = torch.ops.aten.add.Tensor(mul_305, unsqueeze_431);  mul_305 = unsqueeze_431 = None
        convolution_110 = torch.ops.aten.convolution.default(add_200, arg164_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_200 = arg164_1 = None
        view_121 = torch.ops.aten.view.default(convolution_110, [8, 3, 6, 128, -1]);  convolution_110 = None
        permute_46 = torch.ops.aten.permute.default(view_121, [1, 0, 2, 4, 3]);  view_121 = None
        unbind_15 = torch.ops.aten.unbind.int(permute_46);  permute_46 = None
        getitem_45 = unbind_15[0]
        getitem_46 = unbind_15[1]
        getitem_47 = unbind_15[2];  unbind_15 = None
        permute_47 = torch.ops.aten.permute.default(getitem_46, [0, 1, 3, 2]);  getitem_46 = None
        expand_60 = torch.ops.aten.expand.default(getitem_45, [8, 6, 49, 128]);  getitem_45 = None
        clone_155 = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_122 = torch.ops.aten.view.default(clone_155, [48, 49, 128]);  clone_155 = None
        expand_61 = torch.ops.aten.expand.default(permute_47, [8, 6, 128, 49]);  permute_47 = None
        clone_156 = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_123 = torch.ops.aten.view.default(clone_156, [48, 128, 49]);  clone_156 = None
        bmm_30 = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
        view_124 = torch.ops.aten.view.default(bmm_30, [8, 6, 49, 49]);  bmm_30 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_124, 1);  view_124 = None
        amax_default = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, 0.08838834764831845);  sub_tensor = None
        exp_15 = torch.ops.aten.exp.default(mul_tensor_1);  mul_tensor_1 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15 = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        expand_62 = torch.ops.aten.expand.default(div_15, [8, 6, 49, 49]);  div_15 = None
        view_125 = torch.ops.aten.view.default(expand_62, [48, 49, 49]);  expand_62 = None
        expand_63 = torch.ops.aten.expand.default(getitem_47, [8, 6, 49, 128]);  getitem_47 = None
        clone_158 = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_126 = torch.ops.aten.view.default(clone_158, [48, 49, 128]);  clone_158 = None
        bmm_31 = torch.ops.aten.bmm.default(view_125, view_126);  view_125 = view_126 = None
        view_127 = torch.ops.aten.view.default(bmm_31, [8, 6, 49, 128]);  bmm_31 = None
        permute_48 = torch.ops.aten.permute.default(view_127, [0, 1, 3, 2]);  view_127 = None
        clone_159 = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_128 = torch.ops.aten.view.default(clone_159, [8, 768, 7, 7]);  clone_159 = None
        convolution_111 = torch.ops.aten.convolution.default(view_128, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_128 = arg165_1 = None
        add_201 = torch.ops.aten.add.Tensor(add_198, convolution_111);  add_198 = convolution_111 = None
        add_202 = torch.ops.aten.add.Tensor(arg167_1, 1e-05);  arg167_1 = None
        sqrt_54 = torch.ops.aten.sqrt.default(add_202);  add_202 = None
        reciprocal_54 = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_307 = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(mul_307, -1);  mul_307 = None
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_70 = torch.ops.aten.sub.Tensor(add_201, unsqueeze_433);  unsqueeze_433 = None
        mul_308 = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_435);  sub_70 = unsqueeze_435 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_309 = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_437);  mul_308 = unsqueeze_437 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_203 = torch.ops.aten.add.Tensor(mul_309, unsqueeze_439);  mul_309 = unsqueeze_439 = None
        convolution_112 = torch.ops.aten.convolution.default(add_203, arg170_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_203 = arg170_1 = None
        mul_310 = torch.ops.aten.mul.Tensor(convolution_112, 0.5)
        mul_311 = torch.ops.aten.mul.Tensor(convolution_112, 0.7071067811865476);  convolution_112 = None
        erf_43 = torch.ops.aten.erf.default(mul_311);  mul_311 = None
        add_204 = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
        mul_312 = torch.ops.aten.mul.Tensor(mul_310, add_204);  mul_310 = add_204 = None
        convolution_113 = torch.ops.aten.convolution.default(mul_312, arg171_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_312 = arg171_1 = None
        add_205 = torch.ops.aten.add.Tensor(add_201, convolution_113);  add_201 = convolution_113 = None
        add_206 = torch.ops.aten.add.Tensor(arg173_1, 1e-05);  arg173_1 = None
        sqrt_55 = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_55 = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_313 = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(arg172_1, -1);  arg172_1 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(mul_313, -1);  mul_313 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_71 = torch.ops.aten.sub.Tensor(add_205, unsqueeze_441);  add_205 = unsqueeze_441 = None
        mul_314 = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_443);  sub_71 = unsqueeze_443 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_315 = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_445);  mul_314 = unsqueeze_445 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_207 = torch.ops.aten.add.Tensor(mul_315, unsqueeze_447);  mul_315 = unsqueeze_447 = None
        mean_1 = torch.ops.aten.mean.dim(add_207, [-1, -2], True);  add_207 = None
        view_129 = torch.ops.aten.view.default(mean_1, [8, 768]);  mean_1 = None
        permute_49 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg177_1, view_129, permute_49);  arg177_1 = view_129 = permute_49 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 18816, device=device(type='cuda', index=0))
    reader.tensor(buf0, (32, 3, 7, 7), is_leaf=True)  # arg0_1
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
    buf6 = reader.storage(None, 393216, device=device(type='cuda', index=0))
    reader.tensor(buf6, (192, 32, 4, 4), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf7, (192,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf8, (192,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf9, (192,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf10, (192,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf11, (192,), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 602112, device=device(type='cuda', index=0))
    reader.tensor(buf12, (1, 192, 28, 28), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf13, (192,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf14, (192,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf15, (192,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf16, (192,), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf17, (384, 192, 1, 1), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf18, (384, 48, 3, 3), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf19, (192, 384, 1, 1), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf20, (192,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf21, (192,), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf22, (192,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf23, (192,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf24, (384, 192, 1, 1), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf25, (384, 48, 3, 3), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf26, (192, 384, 1, 1), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf27, (192,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf28, (192,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf29, (192,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf30, (192,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf31, (384, 192, 1, 1), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf32, (384, 48, 3, 3), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf33, (192, 384, 1, 1), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf34, (192,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf35, (192,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf36, (192,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf37, (192,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf38, (384, 192, 1, 1), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf39, (384, 48, 3, 3), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf40, (192, 384, 1, 1), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf41, (192,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf42, (192,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf43, (192,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf44, (192,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf45, (384, 192, 1, 1), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf46, (384, 48, 3, 3), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf47, (192, 384, 1, 1), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf48, (192,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf49, (192,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf50, (192,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf51, (192,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf52, (384, 192, 1, 1), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf53, (384, 48, 3, 3), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf54, (192, 384, 1, 1), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf55, (192,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf56, (192,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf57, (192,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf58, (192,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf59, (384, 192, 1, 1), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf60, (384, 48, 3, 3), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf61, (192, 384, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf62, (384, 192, 2, 2), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf63, (384,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf64, (384,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf65, (384,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf66, (384,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf67, (384,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 301056, device=device(type='cuda', index=0))
    reader.tensor(buf68, (1, 384, 14, 14), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf69, (384,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf70, (384,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf71, (384,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf72, (384,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf73, (1152, 384, 1, 1), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf74, (384, 384, 1, 1), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf75, (384,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf76, (384,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf77, (384,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf78, (384,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf79, (1536, 384, 1, 1), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf80, (384, 1536, 1, 1), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf81, (384,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf82, (384,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf83, (384,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf84, (384,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf85, (1152, 384, 1, 1), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf86, (384, 384, 1, 1), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf87, (384,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf88, (384,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf89, (384,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf90, (384,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf91, (1536, 384, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf92, (384, 1536, 1, 1), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf93, (384,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf94, (384,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf95, (384,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf96, (384,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf97, (1152, 384, 1, 1), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf98, (384, 384, 1, 1), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf99, (384,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf100, (384,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf101, (384,), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf102, (384,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf103, (1536, 384, 1, 1), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf104, (384, 1536, 1, 1), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf105, (384,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf106, (384,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf107, (384,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf108, (384,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 1769472, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1152, 384, 1, 1), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf110, (384, 384, 1, 1), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf111, (384,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf112, (384,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf113, (384,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf114, (384,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf115, (1536, 384, 1, 1), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf116, (384, 1536, 1, 1), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 4718592, device=device(type='cuda', index=0))
    reader.tensor(buf117, (768, 384, 2, 2), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf118, (768,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf119, (768,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf120, (768,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf121, (768,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf122, (768,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 150528, device=device(type='cuda', index=0))
    reader.tensor(buf123, (1, 768, 7, 7), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf124, (768,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (768,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf126, (768,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf127, (768,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf128, (2304, 768, 1, 1), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf129, (768, 768, 1, 1), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf130, (768,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf131, (768,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf132, (768,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf133, (768,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf134, (3072, 768, 1, 1), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf135, (768, 3072, 1, 1), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf136, (768,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf137, (768,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf138, (768,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf139, (768,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf140, (2304, 768, 1, 1), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf141, (768, 768, 1, 1), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf142, (768,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf143, (768,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf144, (768,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf145, (768,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf146, (3072, 768, 1, 1), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf147, (768, 3072, 1, 1), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf148, (768,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf149, (768,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf150, (768,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf151, (768,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf152, (2304, 768, 1, 1), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf153, (768, 768, 1, 1), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf154, (768,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf155, (768,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf156, (768,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf157, (768,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf158, (3072, 768, 1, 1), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf159, (768, 3072, 1, 1), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf160, (768,), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf161, (768,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf162, (768,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf163, (768,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 7077888, device=device(type='cuda', index=0))
    reader.tensor(buf164, (2304, 768, 1, 1), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf165, (768, 768, 1, 1), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf166, (768,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf167, (768,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf168, (768,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf169, (768,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf170, (3072, 768, 1, 1), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 9437184, device=device(type='cuda', index=0))
    reader.tensor(buf171, (768, 3072, 1, 1), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf172, (768,), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf173, (768,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf174, (768,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 3072, device=device(type='cuda', index=0))
    reader.tensor(buf175, (768,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 3072000, device=device(type='cuda', index=0))
    reader.tensor(buf176, (1000, 768), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf177, (1000,), is_leaf=True)  # arg177_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)