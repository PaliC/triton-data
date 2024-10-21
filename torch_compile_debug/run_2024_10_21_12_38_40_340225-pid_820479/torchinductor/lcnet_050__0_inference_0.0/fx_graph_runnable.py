
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1):
        convolution_32 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_84 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_27 = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_27 = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_111 = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        sub_27 = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_217);  convolution_32 = unsqueeze_217 = None
        mul_112 = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
        unsqueeze_220 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_221 = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_221);  mul_112 = unsqueeze_221 = None
        unsqueeze_222 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_85 = torch.ops.aten.add.Tensor(mul_113, unsqueeze_223);  mul_113 = unsqueeze_223 = None
        add_86 = torch.ops.aten.add.Tensor(add_85, 3)
        clamp_min_30 = torch.ops.aten.clamp_min.default(add_86, 0);  add_86 = None
        clamp_max_30 = torch.ops.aten.clamp_max.default(clamp_min_30, 6);  clamp_min_30 = None
        mul_114 = torch.ops.aten.mul.Tensor(add_85, clamp_max_30);  add_85 = clamp_max_30 = None
        div_30 = torch.ops.aten.div.Tensor(mul_114, 6);  mul_114 = None
        convolution_33 = torch.ops.aten.convolution.default(div_30, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  div_30 = arg6_1 = None
        add_87 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_28 = torch.ops.aten.sqrt.default(add_87);  add_87 = None
        reciprocal_28 = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_115 = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(mul_115, -1);  mul_115 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        sub_28 = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_225);  convolution_33 = unsqueeze_225 = None
        mul_116 = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_117 = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_229);  mul_116 = unsqueeze_229 = None
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_88 = torch.ops.aten.add.Tensor(mul_117, unsqueeze_231);  mul_117 = unsqueeze_231 = None
        add_89 = torch.ops.aten.add.Tensor(add_88, 3)
        clamp_min_31 = torch.ops.aten.clamp_min.default(add_89, 0);  add_89 = None
        clamp_max_31 = torch.ops.aten.clamp_max.default(clamp_min_31, 6);  clamp_min_31 = None
        mul_118 = torch.ops.aten.mul.Tensor(add_88, clamp_max_31);  add_88 = clamp_max_31 = None
        div_31 = torch.ops.aten.div.Tensor(mul_118, 6);  mul_118 = None
        convolution_34 = torch.ops.aten.convolution.default(div_31, arg11_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_31 = arg11_1 = None
        add_90 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_29 = torch.ops.aten.sqrt.default(add_90);  add_90 = None
        reciprocal_29 = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_119 = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(mul_119, -1);  mul_119 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        sub_29 = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_233);  convolution_34 = unsqueeze_233 = None
        mul_120 = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_121 = torch.ops.aten.mul.Tensor(mul_120, unsqueeze_237);  mul_120 = unsqueeze_237 = None
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_91 = torch.ops.aten.add.Tensor(mul_121, unsqueeze_239);  mul_121 = unsqueeze_239 = None
        add_92 = torch.ops.aten.add.Tensor(add_91, 3)
        clamp_min_32 = torch.ops.aten.clamp_min.default(add_92, 0);  add_92 = None
        clamp_max_32 = torch.ops.aten.clamp_max.default(clamp_min_32, 6);  clamp_min_32 = None
        mul_122 = torch.ops.aten.mul.Tensor(add_91, clamp_max_32);  add_91 = clamp_max_32 = None
        div_32 = torch.ops.aten.div.Tensor(mul_122, 6);  mul_122 = None
        convolution_35 = torch.ops.aten.convolution.default(div_32, arg16_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  div_32 = arg16_1 = None
        add_93 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_30 = torch.ops.aten.sqrt.default(add_93);  add_93 = None
        reciprocal_30 = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_123 = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        sub_30 = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_241);  convolution_35 = unsqueeze_241 = None
        mul_124 = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_245);  mul_124 = unsqueeze_245 = None
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_94 = torch.ops.aten.add.Tensor(mul_125, unsqueeze_247);  mul_125 = unsqueeze_247 = None
        add_95 = torch.ops.aten.add.Tensor(add_94, 3)
        clamp_min_33 = torch.ops.aten.clamp_min.default(add_95, 0);  add_95 = None
        clamp_max_33 = torch.ops.aten.clamp_max.default(clamp_min_33, 6);  clamp_min_33 = None
        mul_126 = torch.ops.aten.mul.Tensor(add_94, clamp_max_33);  add_94 = clamp_max_33 = None
        div_33 = torch.ops.aten.div.Tensor(mul_126, 6);  mul_126 = None
        convolution_36 = torch.ops.aten.convolution.default(div_33, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_33 = arg21_1 = None
        add_96 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_31 = torch.ops.aten.sqrt.default(add_96);  add_96 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_127 = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(mul_127, -1);  mul_127 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        sub_31 = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_249);  convolution_36 = unsqueeze_249 = None
        mul_128 = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_129 = torch.ops.aten.mul.Tensor(mul_128, unsqueeze_253);  mul_128 = unsqueeze_253 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_97 = torch.ops.aten.add.Tensor(mul_129, unsqueeze_255);  mul_129 = unsqueeze_255 = None
        add_98 = torch.ops.aten.add.Tensor(add_97, 3)
        clamp_min_34 = torch.ops.aten.clamp_min.default(add_98, 0);  add_98 = None
        clamp_max_34 = torch.ops.aten.clamp_max.default(clamp_min_34, 6);  clamp_min_34 = None
        mul_130 = torch.ops.aten.mul.Tensor(add_97, clamp_max_34);  add_97 = clamp_max_34 = None
        div_34 = torch.ops.aten.div.Tensor(mul_130, 6);  mul_130 = None
        convolution_37 = torch.ops.aten.convolution.default(div_34, arg26_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 32);  div_34 = arg26_1 = None
        add_99 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_32 = torch.ops.aten.sqrt.default(add_99);  add_99 = None
        reciprocal_32 = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_131 = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(mul_131, -1);  mul_131 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        sub_32 = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_257);  convolution_37 = unsqueeze_257 = None
        mul_132 = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_133 = torch.ops.aten.mul.Tensor(mul_132, unsqueeze_261);  mul_132 = unsqueeze_261 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_100 = torch.ops.aten.add.Tensor(mul_133, unsqueeze_263);  mul_133 = unsqueeze_263 = None
        add_101 = torch.ops.aten.add.Tensor(add_100, 3)
        clamp_min_35 = torch.ops.aten.clamp_min.default(add_101, 0);  add_101 = None
        clamp_max_35 = torch.ops.aten.clamp_max.default(clamp_min_35, 6);  clamp_min_35 = None
        mul_134 = torch.ops.aten.mul.Tensor(add_100, clamp_max_35);  add_100 = clamp_max_35 = None
        div_35 = torch.ops.aten.div.Tensor(mul_134, 6);  mul_134 = None
        convolution_38 = torch.ops.aten.convolution.default(div_35, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_35 = arg31_1 = None
        add_102 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_33 = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_33 = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_135 = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        sub_33 = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_265);  convolution_38 = unsqueeze_265 = None
        mul_136 = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_269);  mul_136 = unsqueeze_269 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_103 = torch.ops.aten.add.Tensor(mul_137, unsqueeze_271);  mul_137 = unsqueeze_271 = None
        add_104 = torch.ops.aten.add.Tensor(add_103, 3)
        clamp_min_36 = torch.ops.aten.clamp_min.default(add_104, 0);  add_104 = None
        clamp_max_36 = torch.ops.aten.clamp_max.default(clamp_min_36, 6);  clamp_min_36 = None
        mul_138 = torch.ops.aten.mul.Tensor(add_103, clamp_max_36);  add_103 = clamp_max_36 = None
        div_36 = torch.ops.aten.div.Tensor(mul_138, 6);  mul_138 = None
        convolution_39 = torch.ops.aten.convolution.default(div_36, arg36_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 32);  div_36 = arg36_1 = None
        add_105 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_34 = torch.ops.aten.sqrt.default(add_105);  add_105 = None
        reciprocal_34 = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_139 = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_272 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(mul_139, -1);  mul_139 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        sub_34 = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_273);  convolution_39 = unsqueeze_273 = None
        mul_140 = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_141 = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_277);  mul_140 = unsqueeze_277 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_106 = torch.ops.aten.add.Tensor(mul_141, unsqueeze_279);  mul_141 = unsqueeze_279 = None
        add_107 = torch.ops.aten.add.Tensor(add_106, 3)
        clamp_min_37 = torch.ops.aten.clamp_min.default(add_107, 0);  add_107 = None
        clamp_max_37 = torch.ops.aten.clamp_max.default(clamp_min_37, 6);  clamp_min_37 = None
        mul_142 = torch.ops.aten.mul.Tensor(add_106, clamp_max_37);  add_106 = clamp_max_37 = None
        div_37 = torch.ops.aten.div.Tensor(mul_142, 6);  mul_142 = None
        convolution_40 = torch.ops.aten.convolution.default(div_37, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_37 = arg41_1 = None
        add_108 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_35 = torch.ops.aten.sqrt.default(add_108);  add_108 = None
        reciprocal_35 = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_143 = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(mul_143, -1);  mul_143 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        sub_35 = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_281);  convolution_40 = unsqueeze_281 = None
        mul_144 = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
        unsqueeze_284 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_145 = torch.ops.aten.mul.Tensor(mul_144, unsqueeze_285);  mul_144 = unsqueeze_285 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_109 = torch.ops.aten.add.Tensor(mul_145, unsqueeze_287);  mul_145 = unsqueeze_287 = None
        add_110 = torch.ops.aten.add.Tensor(add_109, 3)
        clamp_min_38 = torch.ops.aten.clamp_min.default(add_110, 0);  add_110 = None
        clamp_max_38 = torch.ops.aten.clamp_max.default(clamp_min_38, 6);  clamp_min_38 = None
        mul_146 = torch.ops.aten.mul.Tensor(add_109, clamp_max_38);  add_109 = clamp_max_38 = None
        div_38 = torch.ops.aten.div.Tensor(mul_146, 6);  mul_146 = None
        convolution_41 = torch.ops.aten.convolution.default(div_38, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  div_38 = arg46_1 = None
        add_111 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_36 = torch.ops.aten.sqrt.default(add_111);  add_111 = None
        reciprocal_36 = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_147 = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        sub_36 = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_289);  convolution_41 = unsqueeze_289 = None
        mul_148 = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_293);  mul_148 = unsqueeze_293 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_112 = torch.ops.aten.add.Tensor(mul_149, unsqueeze_295);  mul_149 = unsqueeze_295 = None
        add_113 = torch.ops.aten.add.Tensor(add_112, 3)
        clamp_min_39 = torch.ops.aten.clamp_min.default(add_113, 0);  add_113 = None
        clamp_max_39 = torch.ops.aten.clamp_max.default(clamp_min_39, 6);  clamp_min_39 = None
        mul_150 = torch.ops.aten.mul.Tensor(add_112, clamp_max_39);  add_112 = clamp_max_39 = None
        div_39 = torch.ops.aten.div.Tensor(mul_150, 6);  mul_150 = None
        convolution_42 = torch.ops.aten.convolution.default(div_39, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_39 = arg51_1 = None
        add_114 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_37 = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_37 = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_151 = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_296 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(mul_151, -1);  mul_151 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        sub_37 = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_297);  convolution_42 = unsqueeze_297 = None
        mul_152 = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_153 = torch.ops.aten.mul.Tensor(mul_152, unsqueeze_301);  mul_152 = unsqueeze_301 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_115 = torch.ops.aten.add.Tensor(mul_153, unsqueeze_303);  mul_153 = unsqueeze_303 = None
        add_116 = torch.ops.aten.add.Tensor(add_115, 3)
        clamp_min_40 = torch.ops.aten.clamp_min.default(add_116, 0);  add_116 = None
        clamp_max_40 = torch.ops.aten.clamp_max.default(clamp_min_40, 6);  clamp_min_40 = None
        mul_154 = torch.ops.aten.mul.Tensor(add_115, clamp_max_40);  add_115 = clamp_max_40 = None
        div_40 = torch.ops.aten.div.Tensor(mul_154, 6);  mul_154 = None
        convolution_43 = torch.ops.aten.convolution.default(div_40, arg56_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 64);  div_40 = arg56_1 = None
        add_117 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_38 = torch.ops.aten.sqrt.default(add_117);  add_117 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_155 = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(mul_155, -1);  mul_155 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_38 = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_305);  convolution_43 = unsqueeze_305 = None
        mul_156 = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_157 = torch.ops.aten.mul.Tensor(mul_156, unsqueeze_309);  mul_156 = unsqueeze_309 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_118 = torch.ops.aten.add.Tensor(mul_157, unsqueeze_311);  mul_157 = unsqueeze_311 = None
        add_119 = torch.ops.aten.add.Tensor(add_118, 3)
        clamp_min_41 = torch.ops.aten.clamp_min.default(add_119, 0);  add_119 = None
        clamp_max_41 = torch.ops.aten.clamp_max.default(clamp_min_41, 6);  clamp_min_41 = None
        mul_158 = torch.ops.aten.mul.Tensor(add_118, clamp_max_41);  add_118 = clamp_max_41 = None
        div_41 = torch.ops.aten.div.Tensor(mul_158, 6);  mul_158 = None
        convolution_44 = torch.ops.aten.convolution.default(div_41, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_41 = arg61_1 = None
        add_120 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_39 = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_39 = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_159 = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_39 = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_313);  convolution_44 = unsqueeze_313 = None
        mul_160 = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_317);  mul_160 = unsqueeze_317 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_121 = torch.ops.aten.add.Tensor(mul_161, unsqueeze_319);  mul_161 = unsqueeze_319 = None
        add_122 = torch.ops.aten.add.Tensor(add_121, 3)
        clamp_min_42 = torch.ops.aten.clamp_min.default(add_122, 0);  add_122 = None
        clamp_max_42 = torch.ops.aten.clamp_max.default(clamp_min_42, 6);  clamp_min_42 = None
        mul_162 = torch.ops.aten.mul.Tensor(add_121, clamp_max_42);  add_121 = clamp_max_42 = None
        div_42 = torch.ops.aten.div.Tensor(mul_162, 6);  mul_162 = None
        convolution_45 = torch.ops.aten.convolution.default(div_42, arg66_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_42 = arg66_1 = None
        add_123 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_40 = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_40 = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_163 = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(mul_163, -1);  mul_163 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_40 = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_321);  convolution_45 = unsqueeze_321 = None
        mul_164 = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_165 = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_325);  mul_164 = unsqueeze_325 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_124 = torch.ops.aten.add.Tensor(mul_165, unsqueeze_327);  mul_165 = unsqueeze_327 = None
        add_125 = torch.ops.aten.add.Tensor(add_124, 3)
        clamp_min_43 = torch.ops.aten.clamp_min.default(add_125, 0);  add_125 = None
        clamp_max_43 = torch.ops.aten.clamp_max.default(clamp_min_43, 6);  clamp_min_43 = None
        mul_166 = torch.ops.aten.mul.Tensor(add_124, clamp_max_43);  add_124 = clamp_max_43 = None
        div_43 = torch.ops.aten.div.Tensor(mul_166, 6);  mul_166 = None
        convolution_46 = torch.ops.aten.convolution.default(div_43, arg71_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_43 = arg71_1 = None
        add_126 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_126);  add_126 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_167 = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_167, -1);  mul_167 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_41 = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_329);  convolution_46 = unsqueeze_329 = None
        mul_168 = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_169 = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_333);  mul_168 = unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_127 = torch.ops.aten.add.Tensor(mul_169, unsqueeze_335);  mul_169 = unsqueeze_335 = None
        add_128 = torch.ops.aten.add.Tensor(add_127, 3)
        clamp_min_44 = torch.ops.aten.clamp_min.default(add_128, 0);  add_128 = None
        clamp_max_44 = torch.ops.aten.clamp_max.default(clamp_min_44, 6);  clamp_min_44 = None
        mul_170 = torch.ops.aten.mul.Tensor(add_127, clamp_max_44);  add_127 = clamp_max_44 = None
        div_44 = torch.ops.aten.div.Tensor(mul_170, 6);  mul_170 = None
        convolution_47 = torch.ops.aten.convolution.default(div_44, arg76_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_44 = arg76_1 = None
        add_129 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_42 = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_171 = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_42 = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_337);  convolution_47 = unsqueeze_337 = None
        mul_172 = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_341);  mul_172 = unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_130 = torch.ops.aten.add.Tensor(mul_173, unsqueeze_343);  mul_173 = unsqueeze_343 = None
        add_131 = torch.ops.aten.add.Tensor(add_130, 3)
        clamp_min_45 = torch.ops.aten.clamp_min.default(add_131, 0);  add_131 = None
        clamp_max_45 = torch.ops.aten.clamp_max.default(clamp_min_45, 6);  clamp_min_45 = None
        mul_174 = torch.ops.aten.mul.Tensor(add_130, clamp_max_45);  add_130 = clamp_max_45 = None
        div_45 = torch.ops.aten.div.Tensor(mul_174, 6);  mul_174 = None
        convolution_48 = torch.ops.aten.convolution.default(div_45, arg81_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_45 = arg81_1 = None
        add_132 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_43 = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_175 = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(mul_175, -1);  mul_175 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_43 = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_345);  convolution_48 = unsqueeze_345 = None
        mul_176 = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_349);  mul_176 = unsqueeze_349 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_133 = torch.ops.aten.add.Tensor(mul_177, unsqueeze_351);  mul_177 = unsqueeze_351 = None
        add_134 = torch.ops.aten.add.Tensor(add_133, 3)
        clamp_min_46 = torch.ops.aten.clamp_min.default(add_134, 0);  add_134 = None
        clamp_max_46 = torch.ops.aten.clamp_max.default(clamp_min_46, 6);  clamp_min_46 = None
        mul_178 = torch.ops.aten.mul.Tensor(add_133, clamp_max_46);  add_133 = clamp_max_46 = None
        div_46 = torch.ops.aten.div.Tensor(mul_178, 6);  mul_178 = None
        convolution_49 = torch.ops.aten.convolution.default(div_46, arg86_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_46 = arg86_1 = None
        add_135 = torch.ops.aten.add.Tensor(arg88_1, 1e-05);  arg88_1 = None
        sqrt_44 = torch.ops.aten.sqrt.default(add_135);  add_135 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_179 = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(arg87_1, -1);  arg87_1 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_179, -1);  mul_179 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_44 = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_353);  convolution_49 = unsqueeze_353 = None
        mul_180 = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_181 = torch.ops.aten.mul.Tensor(mul_180, unsqueeze_357);  mul_180 = unsqueeze_357 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_136 = torch.ops.aten.add.Tensor(mul_181, unsqueeze_359);  mul_181 = unsqueeze_359 = None
        add_137 = torch.ops.aten.add.Tensor(add_136, 3)
        clamp_min_47 = torch.ops.aten.clamp_min.default(add_137, 0);  add_137 = None
        clamp_max_47 = torch.ops.aten.clamp_max.default(clamp_min_47, 6);  clamp_min_47 = None
        mul_182 = torch.ops.aten.mul.Tensor(add_136, clamp_max_47);  add_136 = clamp_max_47 = None
        div_47 = torch.ops.aten.div.Tensor(mul_182, 6);  mul_182 = None
        convolution_50 = torch.ops.aten.convolution.default(div_47, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_47 = arg91_1 = None
        add_138 = torch.ops.aten.add.Tensor(arg93_1, 1e-05);  arg93_1 = None
        sqrt_45 = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_183 = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(arg92_1, -1);  arg92_1 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_361);  convolution_50 = unsqueeze_361 = None
        mul_184 = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_365);  mul_184 = unsqueeze_365 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_139 = torch.ops.aten.add.Tensor(mul_185, unsqueeze_367);  mul_185 = unsqueeze_367 = None
        add_140 = torch.ops.aten.add.Tensor(add_139, 3)
        clamp_min_48 = torch.ops.aten.clamp_min.default(add_140, 0);  add_140 = None
        clamp_max_48 = torch.ops.aten.clamp_max.default(clamp_min_48, 6);  clamp_min_48 = None
        mul_186 = torch.ops.aten.mul.Tensor(add_139, clamp_max_48);  add_139 = clamp_max_48 = None
        div_48 = torch.ops.aten.div.Tensor(mul_186, 6);  mul_186 = None
        convolution_51 = torch.ops.aten.convolution.default(div_48, arg96_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_48 = arg96_1 = None
        add_141 = torch.ops.aten.add.Tensor(arg98_1, 1e-05);  arg98_1 = None
        sqrt_46 = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_46 = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_187 = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(arg97_1, -1);  arg97_1 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(mul_187, -1);  mul_187 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_46 = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_369);  convolution_51 = unsqueeze_369 = None
        mul_188 = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, unsqueeze_373);  mul_188 = unsqueeze_373 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_142 = torch.ops.aten.add.Tensor(mul_189, unsqueeze_375);  mul_189 = unsqueeze_375 = None
        add_143 = torch.ops.aten.add.Tensor(add_142, 3)
        clamp_min_49 = torch.ops.aten.clamp_min.default(add_143, 0);  add_143 = None
        clamp_max_49 = torch.ops.aten.clamp_max.default(clamp_min_49, 6);  clamp_min_49 = None
        mul_190 = torch.ops.aten.mul.Tensor(add_142, clamp_max_49);  add_142 = clamp_max_49 = None
        div_49 = torch.ops.aten.div.Tensor(mul_190, 6);  mul_190 = None
        convolution_52 = torch.ops.aten.convolution.default(div_49, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_49 = arg101_1 = None
        add_144 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_47 = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_47 = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_191 = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(mul_191, -1);  mul_191 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_47 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_377);  convolution_52 = unsqueeze_377 = None
        mul_192 = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_381);  mul_192 = unsqueeze_381 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_145 = torch.ops.aten.add.Tensor(mul_193, unsqueeze_383);  mul_193 = unsqueeze_383 = None
        add_146 = torch.ops.aten.add.Tensor(add_145, 3)
        clamp_min_50 = torch.ops.aten.clamp_min.default(add_146, 0);  add_146 = None
        clamp_max_50 = torch.ops.aten.clamp_max.default(clamp_min_50, 6);  clamp_min_50 = None
        mul_194 = torch.ops.aten.mul.Tensor(add_145, clamp_max_50);  add_145 = clamp_max_50 = None
        div_50 = torch.ops.aten.div.Tensor(mul_194, 6);  mul_194 = None
        convolution_53 = torch.ops.aten.convolution.default(div_50, arg106_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 128);  div_50 = arg106_1 = None
        add_147 = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_48 = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_48 = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_195 = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_48 = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_385);  convolution_53 = unsqueeze_385 = None
        mul_196 = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_389);  mul_196 = unsqueeze_389 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_148 = torch.ops.aten.add.Tensor(mul_197, unsqueeze_391);  mul_197 = unsqueeze_391 = None
        add_149 = torch.ops.aten.add.Tensor(add_148, 3)
        clamp_min_51 = torch.ops.aten.clamp_min.default(add_149, 0);  add_149 = None
        clamp_max_51 = torch.ops.aten.clamp_max.default(clamp_min_51, 6);  clamp_min_51 = None
        mul_198 = torch.ops.aten.mul.Tensor(add_148, clamp_max_51);  add_148 = clamp_max_51 = None
        div_51 = torch.ops.aten.div.Tensor(mul_198, 6);  mul_198 = None
        convolution_54 = torch.ops.aten.convolution.default(div_51, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  div_51 = arg111_1 = None
        add_150 = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_49 = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_49 = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_199 = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(mul_199, -1);  mul_199 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_49 = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_393);  convolution_54 = unsqueeze_393 = None
        mul_200 = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_201 = torch.ops.aten.mul.Tensor(mul_200, unsqueeze_397);  mul_200 = unsqueeze_397 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_151 = torch.ops.aten.add.Tensor(mul_201, unsqueeze_399);  mul_201 = unsqueeze_399 = None
        add_152 = torch.ops.aten.add.Tensor(add_151, 3)
        clamp_min_52 = torch.ops.aten.clamp_min.default(add_152, 0);  add_152 = None
        clamp_max_52 = torch.ops.aten.clamp_max.default(clamp_min_52, 6);  clamp_min_52 = None
        mul_202 = torch.ops.aten.mul.Tensor(add_151, clamp_max_52);  add_151 = clamp_max_52 = None
        div_52 = torch.ops.aten.div.Tensor(mul_202, 6);  mul_202 = None
        convolution_55 = torch.ops.aten.convolution.default(div_52, arg116_1, None, [2, 2], [2, 2], [1, 1], False, [0, 0], 128);  div_52 = arg116_1 = None
        add_153 = torch.ops.aten.add.Tensor(arg118_1, 1e-05);  arg118_1 = None
        sqrt_50 = torch.ops.aten.sqrt.default(add_153);  add_153 = None
        reciprocal_50 = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_203 = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_203, -1);  mul_203 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_50 = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_401);  convolution_55 = unsqueeze_401 = None
        mul_204 = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_205 = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_405);  mul_204 = unsqueeze_405 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_154 = torch.ops.aten.add.Tensor(mul_205, unsqueeze_407);  mul_205 = unsqueeze_407 = None
        add_155 = torch.ops.aten.add.Tensor(add_154, 3)
        clamp_min_53 = torch.ops.aten.clamp_min.default(add_155, 0);  add_155 = None
        clamp_max_53 = torch.ops.aten.clamp_max.default(clamp_min_53, 6);  clamp_min_53 = None
        mul_206 = torch.ops.aten.mul.Tensor(add_154, clamp_max_53);  add_154 = clamp_max_53 = None
        div_53 = torch.ops.aten.div.Tensor(mul_206, 6);  mul_206 = None
        mean_3 = torch.ops.aten.mean.dim(div_53, [2, 3], True)
        convolution_56 = torch.ops.aten.convolution.default(mean_3, arg121_1, arg122_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_3 = arg121_1 = arg122_1 = None
        relu_2 = torch.ops.aten.relu.default(convolution_56);  convolution_56 = None
        convolution_57 = torch.ops.aten.convolution.default(relu_2, arg123_1, arg124_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg123_1 = arg124_1 = None
        add_156 = torch.ops.aten.add.Tensor(convolution_57, 3);  convolution_57 = None
        clamp_min_54 = torch.ops.aten.clamp_min.default(add_156, 0);  add_156 = None
        clamp_max_54 = torch.ops.aten.clamp_max.default(clamp_min_54, 6);  clamp_min_54 = None
        div_54 = torch.ops.aten.div.Tensor(clamp_max_54, 6);  clamp_max_54 = None
        mul_207 = torch.ops.aten.mul.Tensor(div_53, div_54);  div_53 = div_54 = None
        convolution_58 = torch.ops.aten.convolution.default(mul_207, arg125_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_207 = arg125_1 = None
        add_157 = torch.ops.aten.add.Tensor(arg127_1, 1e-05);  arg127_1 = None
        sqrt_51 = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_51 = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_208 = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(mul_208, -1);  mul_208 = None
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_51 = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_409);  convolution_58 = unsqueeze_409 = None
        mul_209 = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_210 = torch.ops.aten.mul.Tensor(mul_209, unsqueeze_413);  mul_209 = unsqueeze_413 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_158 = torch.ops.aten.add.Tensor(mul_210, unsqueeze_415);  mul_210 = unsqueeze_415 = None
        add_159 = torch.ops.aten.add.Tensor(add_158, 3)
        clamp_min_55 = torch.ops.aten.clamp_min.default(add_159, 0);  add_159 = None
        clamp_max_55 = torch.ops.aten.clamp_max.default(clamp_min_55, 6);  clamp_min_55 = None
        mul_211 = torch.ops.aten.mul.Tensor(add_158, clamp_max_55);  add_158 = clamp_max_55 = None
        div_55 = torch.ops.aten.div.Tensor(mul_211, 6);  mul_211 = None
        convolution_59 = torch.ops.aten.convolution.default(div_55, arg130_1, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 256);  div_55 = arg130_1 = None
        add_160 = torch.ops.aten.add.Tensor(arg132_1, 1e-05);  arg132_1 = None
        sqrt_52 = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_212 = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(mul_212, -1);  mul_212 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_417);  convolution_59 = unsqueeze_417 = None
        mul_213 = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_214 = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_421);  mul_213 = unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_161 = torch.ops.aten.add.Tensor(mul_214, unsqueeze_423);  mul_214 = unsqueeze_423 = None
        add_162 = torch.ops.aten.add.Tensor(add_161, 3)
        clamp_min_56 = torch.ops.aten.clamp_min.default(add_162, 0);  add_162 = None
        clamp_max_56 = torch.ops.aten.clamp_max.default(clamp_min_56, 6);  clamp_min_56 = None
        mul_215 = torch.ops.aten.mul.Tensor(add_161, clamp_max_56);  add_161 = clamp_max_56 = None
        div_56 = torch.ops.aten.div.Tensor(mul_215, 6);  mul_215 = None
        mean_4 = torch.ops.aten.mean.dim(div_56, [2, 3], True)
        convolution_60 = torch.ops.aten.convolution.default(mean_4, arg135_1, arg136_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_4 = arg135_1 = arg136_1 = None
        relu_3 = torch.ops.aten.relu.default(convolution_60);  convolution_60 = None
        convolution_61 = torch.ops.aten.convolution.default(relu_3, arg137_1, arg138_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_3 = arg137_1 = arg138_1 = None
        add_163 = torch.ops.aten.add.Tensor(convolution_61, 3);  convolution_61 = None
        clamp_min_57 = torch.ops.aten.clamp_min.default(add_163, 0);  add_163 = None
        clamp_max_57 = torch.ops.aten.clamp_max.default(clamp_min_57, 6);  clamp_min_57 = None
        div_57 = torch.ops.aten.div.Tensor(clamp_max_57, 6);  clamp_max_57 = None
        mul_216 = torch.ops.aten.mul.Tensor(div_56, div_57);  div_56 = div_57 = None
        convolution_62 = torch.ops.aten.convolution.default(mul_216, arg139_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_216 = arg139_1 = None
        add_164 = torch.ops.aten.add.Tensor(arg141_1, 1e-05);  arg141_1 = None
        sqrt_53 = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_53 = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_217 = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_217, -1);  mul_217 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_53 = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_425);  convolution_62 = unsqueeze_425 = None
        mul_218 = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_429);  mul_218 = unsqueeze_429 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_165 = torch.ops.aten.add.Tensor(mul_219, unsqueeze_431);  mul_219 = unsqueeze_431 = None
        add_166 = torch.ops.aten.add.Tensor(add_165, 3)
        clamp_min_58 = torch.ops.aten.clamp_min.default(add_166, 0);  add_166 = None
        clamp_max_58 = torch.ops.aten.clamp_max.default(clamp_min_58, 6);  clamp_min_58 = None
        mul_220 = torch.ops.aten.mul.Tensor(add_165, clamp_max_58);  add_165 = clamp_max_58 = None
        div_58 = torch.ops.aten.div.Tensor(mul_220, 6);  mul_220 = None
        mean_5 = torch.ops.aten.mean.dim(div_58, [-1, -2], True);  div_58 = None
        convolution_63 = torch.ops.aten.convolution.default(mean_5, arg144_1, arg145_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mean_5 = arg144_1 = arg145_1 = None
        add_167 = torch.ops.aten.add.Tensor(convolution_63, 3)
        clamp_min_59 = torch.ops.aten.clamp_min.default(add_167, 0);  add_167 = None
        clamp_max_59 = torch.ops.aten.clamp_max.default(clamp_min_59, 6);  clamp_min_59 = None
        mul_221 = torch.ops.aten.mul.Tensor(convolution_63, clamp_max_59);  convolution_63 = clamp_max_59 = None
        div_59 = torch.ops.aten.div.Tensor(mul_221, 6);  mul_221 = None
        permute_1 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        view_3 = torch.ops.aten.view.default(div_59, [8, 1280]);  div_59 = None
        addmm_1 = torch.ops.aten.addmm.default(arg147_1, view_3, permute_1);  arg147_1 = view_3 = permute_1 = None
        return (addmm_1,)
        
def load_args(reader):
    buf0 = reader.storage(None, 864, device=device(type='cuda', index=0))
    reader.tensor(buf0, (8, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4816896, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 224, 224), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf2, (8,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf3, (8,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf4, (8,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf5, (8,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 288, device=device(type='cuda', index=0))
    reader.tensor(buf6, (8, 1, 3, 3), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf7, (8,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf8, (8,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf9, (8,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 32, device=device(type='cuda', index=0))
    reader.tensor(buf10, (8,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf11, (16, 8, 1, 1), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf12, (16,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf13, (16,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf14, (16,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf15, (16,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf16, (16, 1, 3, 3), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf17, (16,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf18, (16,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf19, (16,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf20, (16,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf21, (32, 16, 1, 1), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf22, (32,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf23, (32,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf24, (32,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf25, (32,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf26, (32, 1, 3, 3), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf27, (32,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf28, (32,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf29, (32,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf30, (32,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf31, (32, 32, 1, 1), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf32, (32,), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf33, (32,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf34, (32,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf35, (32,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf36, (32, 1, 3, 3), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf37, (32,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf38, (32,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf39, (32,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf40, (32,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf41, (64, 32, 1, 1), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf42, (64,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf43, (64,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64, 1, 3, 3), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf50, (64,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf51, (64, 64, 1, 1), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf52, (64,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf53, (64,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf54, (64,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf55, (64,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf56, (64, 1, 3, 3), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf57, (64,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf58, (64,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf59, (64,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf60, (64,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 32768, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128, 64, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf63, (128,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf64, (128,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf65, (128,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 12800, device=device(type='cuda', index=0))
    reader.tensor(buf66, (128, 1, 5, 5), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf67, (128,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf68, (128,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf69, (128,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf70, (128,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf71, (128, 128, 1, 1), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf72, (128,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf73, (128,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 12800, device=device(type='cuda', index=0))
    reader.tensor(buf76, (128, 1, 5, 5), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf77, (128,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf78, (128,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf79, (128,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf80, (128,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf81, (128, 128, 1, 1), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf82, (128,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf84, (128,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf85, (128,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 12800, device=device(type='cuda', index=0))
    reader.tensor(buf86, (128, 1, 5, 5), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf87, (128,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf88, (128,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf89, (128,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf90, (128,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf91, (128, 128, 1, 1), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf92, (128,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf93, (128,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf94, (128,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf95, (128,), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 12800, device=device(type='cuda', index=0))
    reader.tensor(buf96, (128, 1, 5, 5), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf97, (128,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf98, (128,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf99, (128,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf100, (128,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf101, (128, 128, 1, 1), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf102, (128,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf103, (128,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf104, (128,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf105, (128,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 12800, device=device(type='cuda', index=0))
    reader.tensor(buf106, (128, 1, 5, 5), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf107, (128,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf108, (128,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf109, (128,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf110, (128,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf111, (128, 128, 1, 1), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf112, (128,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf113, (128,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf114, (128,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf115, (128,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 12800, device=device(type='cuda', index=0))
    reader.tensor(buf116, (128, 1, 5, 5), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf117, (128,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf118, (128,), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf119, (128,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf120, (128,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf121, (32, 128, 1, 1), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf122, (32,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf123, (128, 32, 1, 1), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf124, (128,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf125, (256, 128, 1, 1), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf126, (256,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf127, (256,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf128, (256,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf129, (256,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 25600, device=device(type='cuda', index=0))
    reader.tensor(buf130, (256, 1, 5, 5), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf131, (256,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf132, (256,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf133, (256,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf134, (256,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf135, (64, 256, 1, 1), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf136, (64,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf137, (256, 64, 1, 1), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf138, (256,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf139, (256, 256, 1, 1), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf140, (256,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf141, (256,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf142, (256,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf143, (256,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 1310720, device=device(type='cuda', index=0))
    reader.tensor(buf144, (1280, 256, 1, 1), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 5120, device=device(type='cuda', index=0))
    reader.tensor(buf145, (1280,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 5120000, device=device(type='cuda', index=0))
    reader.tensor(buf146, (1000, 1280), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf147, (1000,), is_leaf=True)  # arg147_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)