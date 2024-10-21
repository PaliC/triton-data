
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1):
        convolution_35 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_126 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_32 = torch.ops.aten.sqrt.default(add_126);  add_126 = None
        reciprocal_32 = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_172 = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(mul_172, -1);  mul_172 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        sub_53 = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_257);  convolution_35 = unsqueeze_257 = None
        mul_173 = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_259);  sub_53 = unsqueeze_259 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_174 = torch.ops.aten.mul.Tensor(mul_173, unsqueeze_261);  mul_173 = unsqueeze_261 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_127 = torch.ops.aten.add.Tensor(mul_174, unsqueeze_263);  mul_174 = unsqueeze_263 = None
        sigmoid_34 = torch.ops.aten.sigmoid.default(add_127)
        mul_175 = torch.ops.aten.mul.Tensor(add_127, sigmoid_34);  add_127 = sigmoid_34 = None
        convolution_36 = torch.ops.aten.convolution.default(mul_175, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_175 = arg6_1 = None
        add_128 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_33 = torch.ops.aten.sqrt.default(add_128);  add_128 = None
        reciprocal_33 = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_176 = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(mul_176, -1);  mul_176 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        sub_54 = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_265);  convolution_36 = unsqueeze_265 = None
        mul_177 = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_267);  sub_54 = unsqueeze_267 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_178 = torch.ops.aten.mul.Tensor(mul_177, unsqueeze_269);  mul_177 = unsqueeze_269 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_129 = torch.ops.aten.add.Tensor(mul_178, unsqueeze_271);  mul_178 = unsqueeze_271 = None
        sigmoid_35 = torch.ops.aten.sigmoid.default(add_129)
        mul_179 = torch.ops.aten.mul.Tensor(add_129, sigmoid_35);  add_129 = sigmoid_35 = None
        convolution_37 = torch.ops.aten.convolution.default(mul_179, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 64);  mul_179 = arg11_1 = None
        add_130 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_34 = torch.ops.aten.sqrt.default(add_130);  add_130 = None
        reciprocal_34 = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_180 = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_272 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        sub_55 = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_273);  convolution_37 = unsqueeze_273 = None
        mul_181 = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_275);  sub_55 = unsqueeze_275 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_182 = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_277);  mul_181 = unsqueeze_277 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_131 = torch.ops.aten.add.Tensor(mul_182, unsqueeze_279);  mul_182 = unsqueeze_279 = None
        sigmoid_36 = torch.ops.aten.sigmoid.default(add_131)
        mul_183 = torch.ops.aten.mul.Tensor(add_131, sigmoid_36);  add_131 = sigmoid_36 = None
        convolution_38 = torch.ops.aten.convolution.default(mul_183, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_183 = arg16_1 = None
        add_132 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_35 = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_35 = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_184 = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(mul_184, -1);  mul_184 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        sub_56 = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_281);  convolution_38 = unsqueeze_281 = None
        mul_185 = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_283);  sub_56 = unsqueeze_283 = None
        unsqueeze_284 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_186 = torch.ops.aten.mul.Tensor(mul_185, unsqueeze_285);  mul_185 = unsqueeze_285 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_133 = torch.ops.aten.add.Tensor(mul_186, unsqueeze_287);  mul_186 = unsqueeze_287 = None
        convolution_39 = torch.ops.aten.convolution.default(add_133, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_133 = arg21_1 = None
        add_134 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_36 = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_36 = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_187 = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(mul_187, -1);  mul_187 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        sub_57 = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_289);  convolution_39 = unsqueeze_289 = None
        mul_188 = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_291);  sub_57 = unsqueeze_291 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_189 = torch.ops.aten.mul.Tensor(mul_188, unsqueeze_293);  mul_188 = unsqueeze_293 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_135 = torch.ops.aten.add.Tensor(mul_189, unsqueeze_295);  mul_189 = unsqueeze_295 = None
        sigmoid_37 = torch.ops.aten.sigmoid.default(add_135)
        mul_190 = torch.ops.aten.mul.Tensor(add_135, sigmoid_37);  add_135 = sigmoid_37 = None
        convolution_40 = torch.ops.aten.convolution.default(mul_190, arg26_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 128);  mul_190 = arg26_1 = None
        add_136 = torch.ops.aten.add.Tensor(arg28_1, 1e-05);  arg28_1 = None
        sqrt_37 = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_37 = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_191 = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_296 = torch.ops.aten.unsqueeze.default(arg27_1, -1);  arg27_1 = None
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(mul_191, -1);  mul_191 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        sub_58 = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_297);  convolution_40 = unsqueeze_297 = None
        mul_192 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_299);  sub_58 = unsqueeze_299 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, unsqueeze_301);  mul_192 = unsqueeze_301 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_137 = torch.ops.aten.add.Tensor(mul_193, unsqueeze_303);  mul_193 = unsqueeze_303 = None
        sigmoid_38 = torch.ops.aten.sigmoid.default(add_137)
        mul_194 = torch.ops.aten.mul.Tensor(add_137, sigmoid_38);  add_137 = sigmoid_38 = None
        convolution_41 = torch.ops.aten.convolution.default(mul_194, arg31_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_194 = arg31_1 = None
        add_138 = torch.ops.aten.add.Tensor(arg33_1, 1e-05);  arg33_1 = None
        sqrt_38 = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_195 = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(arg32_1, -1);  arg32_1 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_59 = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_305);  convolution_41 = unsqueeze_305 = None
        mul_196 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_307);  sub_59 = unsqueeze_307 = None
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_309);  mul_196 = unsqueeze_309 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_139 = torch.ops.aten.add.Tensor(mul_197, unsqueeze_311);  mul_197 = unsqueeze_311 = None
        convolution_42 = torch.ops.aten.convolution.default(add_139, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg36_1 = None
        add_140 = torch.ops.aten.add.Tensor(arg38_1, 1e-05);  arg38_1 = None
        sqrt_39 = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_39 = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_198 = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(arg37_1, -1);  arg37_1 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_60 = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_313);  convolution_42 = unsqueeze_313 = None
        mul_199 = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_315);  sub_60 = unsqueeze_315 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_317);  mul_199 = unsqueeze_317 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_141 = torch.ops.aten.add.Tensor(mul_200, unsqueeze_319);  mul_200 = unsqueeze_319 = None
        sigmoid_39 = torch.ops.aten.sigmoid.default(add_141)
        mul_201 = torch.ops.aten.mul.Tensor(add_141, sigmoid_39);  add_141 = sigmoid_39 = None
        convolution_43 = torch.ops.aten.convolution.default(mul_201, arg41_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256);  mul_201 = arg41_1 = None
        add_142 = torch.ops.aten.add.Tensor(arg43_1, 1e-05);  arg43_1 = None
        sqrt_40 = torch.ops.aten.sqrt.default(add_142);  add_142 = None
        reciprocal_40 = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_202 = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(arg42_1, -1);  arg42_1 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(mul_202, -1);  mul_202 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_61 = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_321);  convolution_43 = unsqueeze_321 = None
        mul_203 = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_323);  sub_61 = unsqueeze_323 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_204 = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_325);  mul_203 = unsqueeze_325 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_143 = torch.ops.aten.add.Tensor(mul_204, unsqueeze_327);  mul_204 = unsqueeze_327 = None
        sigmoid_40 = torch.ops.aten.sigmoid.default(add_143)
        mul_205 = torch.ops.aten.mul.Tensor(add_143, sigmoid_40);  add_143 = sigmoid_40 = None
        convolution_44 = torch.ops.aten.convolution.default(mul_205, arg46_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_205 = arg46_1 = None
        add_144 = torch.ops.aten.add.Tensor(arg48_1, 1e-05);  arg48_1 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_144);  add_144 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_206 = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(arg47_1, -1);  arg47_1 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_206, -1);  mul_206 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_62 = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_329);  convolution_44 = unsqueeze_329 = None
        mul_207 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_331);  sub_62 = unsqueeze_331 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_208 = torch.ops.aten.mul.Tensor(mul_207, unsqueeze_333);  mul_207 = unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_145 = torch.ops.aten.add.Tensor(mul_208, unsqueeze_335);  mul_208 = unsqueeze_335 = None
        add_146 = torch.ops.aten.add.Tensor(add_145, add_139);  add_145 = add_139 = None
        convolution_45 = torch.ops.aten.convolution.default(add_146, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg51_1 = None
        add_147 = torch.ops.aten.add.Tensor(arg53_1, 1e-05);  arg53_1 = None
        sqrt_42 = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_209 = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_209, -1);  mul_209 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_63 = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_337);  convolution_45 = unsqueeze_337 = None
        mul_210 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_339);  sub_63 = unsqueeze_339 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_211 = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_341);  mul_210 = unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_148 = torch.ops.aten.add.Tensor(mul_211, unsqueeze_343);  mul_211 = unsqueeze_343 = None
        sigmoid_41 = torch.ops.aten.sigmoid.default(add_148)
        mul_212 = torch.ops.aten.mul.Tensor(add_148, sigmoid_41);  add_148 = sigmoid_41 = None
        convolution_46 = torch.ops.aten.convolution.default(mul_212, arg56_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 256);  mul_212 = arg56_1 = None
        add_149 = torch.ops.aten.add.Tensor(arg58_1, 1e-05);  arg58_1 = None
        sqrt_43 = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_213 = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_64 = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_345);  convolution_46 = unsqueeze_345 = None
        mul_214 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_347);  sub_64 = unsqueeze_347 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_215 = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_349);  mul_214 = unsqueeze_349 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_150 = torch.ops.aten.add.Tensor(mul_215, unsqueeze_351);  mul_215 = unsqueeze_351 = None
        sigmoid_42 = torch.ops.aten.sigmoid.default(add_150)
        mul_216 = torch.ops.aten.mul.Tensor(add_150, sigmoid_42);  add_150 = sigmoid_42 = None
        convolution_47 = torch.ops.aten.convolution.default(mul_216, arg61_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_216 = arg61_1 = None
        add_151 = torch.ops.aten.add.Tensor(arg63_1, 1e-05);  arg63_1 = None
        sqrt_44 = torch.ops.aten.sqrt.default(add_151);  add_151 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_217 = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_217, -1);  mul_217 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_65 = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_353);  convolution_47 = unsqueeze_353 = None
        mul_218 = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_355);  sub_65 = unsqueeze_355 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_219 = torch.ops.aten.mul.Tensor(mul_218, unsqueeze_357);  mul_218 = unsqueeze_357 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_152 = torch.ops.aten.add.Tensor(mul_219, unsqueeze_359);  mul_219 = unsqueeze_359 = None
        add_153 = torch.ops.aten.add.Tensor(add_152, add_146);  add_152 = add_146 = None
        convolution_48 = torch.ops.aten.convolution.default(add_153, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  add_153 = arg66_1 = None
        add_154 = torch.ops.aten.add.Tensor(arg68_1, 1e-05);  arg68_1 = None
        sqrt_45 = torch.ops.aten.sqrt.default(add_154);  add_154 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_220 = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_220, -1);  mul_220 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_66 = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_361);  convolution_48 = unsqueeze_361 = None
        mul_221 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_363);  sub_66 = unsqueeze_363 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_222 = torch.ops.aten.mul.Tensor(mul_221, unsqueeze_365);  mul_221 = unsqueeze_365 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_155 = torch.ops.aten.add.Tensor(mul_222, unsqueeze_367);  mul_222 = unsqueeze_367 = None
        sigmoid_43 = torch.ops.aten.sigmoid.default(add_155)
        mul_223 = torch.ops.aten.mul.Tensor(add_155, sigmoid_43);  add_155 = sigmoid_43 = None
        convolution_49 = torch.ops.aten.convolution.default(mul_223, arg71_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 256);  mul_223 = arg71_1 = None
        add_156 = torch.ops.aten.add.Tensor(arg73_1, 1e-05);  arg73_1 = None
        sqrt_46 = torch.ops.aten.sqrt.default(add_156);  add_156 = None
        reciprocal_46 = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_224 = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(mul_224, -1);  mul_224 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_369);  convolution_49 = unsqueeze_369 = None
        mul_225 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_371);  sub_67 = unsqueeze_371 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_226 = torch.ops.aten.mul.Tensor(mul_225, unsqueeze_373);  mul_225 = unsqueeze_373 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_157 = torch.ops.aten.add.Tensor(mul_226, unsqueeze_375);  mul_226 = unsqueeze_375 = None
        sigmoid_44 = torch.ops.aten.sigmoid.default(add_157)
        mul_227 = torch.ops.aten.mul.Tensor(add_157, sigmoid_44);  add_157 = sigmoid_44 = None
        convolution_50 = torch.ops.aten.convolution.default(mul_227, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_227 = arg76_1 = None
        add_158 = torch.ops.aten.add.Tensor(arg78_1, 1e-05);  arg78_1 = None
        sqrt_47 = torch.ops.aten.sqrt.default(add_158);  add_158 = None
        reciprocal_47 = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_228 = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_68 = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_377);  convolution_50 = unsqueeze_377 = None
        mul_229 = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_379);  sub_68 = unsqueeze_379 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_381);  mul_229 = unsqueeze_381 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_159 = torch.ops.aten.add.Tensor(mul_230, unsqueeze_383);  mul_230 = unsqueeze_383 = None
        convolution_51 = torch.ops.aten.convolution.default(add_159, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg81_1 = None
        add_160 = torch.ops.aten.add.Tensor(arg83_1, 1e-05);  arg83_1 = None
        sqrt_48 = torch.ops.aten.sqrt.default(add_160);  add_160 = None
        reciprocal_48 = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_231 = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_69 = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_385);  convolution_51 = unsqueeze_385 = None
        mul_232 = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_387);  sub_69 = unsqueeze_387 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_389);  mul_232 = unsqueeze_389 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_161 = torch.ops.aten.add.Tensor(mul_233, unsqueeze_391);  mul_233 = unsqueeze_391 = None
        sigmoid_45 = torch.ops.aten.sigmoid.default(add_161)
        mul_234 = torch.ops.aten.mul.Tensor(add_161, sigmoid_45);  add_161 = sigmoid_45 = None
        convolution_52 = torch.ops.aten.convolution.default(mul_234, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_234 = arg86_1 = None
        view_109 = torch.ops.aten.view.default(convolution_52, [18432, 2, 16, 2]);  convolution_52 = None
        permute_67 = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        clone_40 = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_110 = torch.ops.aten.view.default(clone_40, [8, 144, 256, 4]);  clone_40 = None
        permute_68 = torch.ops.aten.permute.default(view_110, [0, 3, 2, 1]);  view_110 = None
        clone_41 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_111 = torch.ops.aten.view.default(clone_41, [32, 256, 144]);  clone_41 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(view_111, [2], correction = 0, keepdim = True)
        getitem_105 = var_mean_21[0]
        getitem_106 = var_mean_21[1];  var_mean_21 = None
        add_162 = torch.ops.aten.add.Tensor(getitem_105, 1e-05);  getitem_105 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_70 = torch.ops.aten.sub.Tensor(view_111, getitem_106);  getitem_106 = None
        mul_235 = torch.ops.aten.mul.Tensor(sub_70, rsqrt_21);  sub_70 = rsqrt_21 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_235, arg87_1);  mul_235 = arg87_1 = None
        add_163 = torch.ops.aten.add.Tensor(mul_236, arg88_1);  mul_236 = arg88_1 = None
        view_112 = torch.ops.aten.view.default(add_163, [8192, 144]);  add_163 = None
        permute_69 = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_37 = torch.ops.aten.addmm.default(arg90_1, view_112, permute_69);  arg90_1 = view_112 = permute_69 = None
        view_113 = torch.ops.aten.view.default(addmm_37, [32, 256, 432]);  addmm_37 = None
        view_114 = torch.ops.aten.view.default(view_113, [32, 256, 3, 4, 36]);  view_113 = None
        permute_70 = torch.ops.aten.permute.default(view_114, [2, 0, 3, 1, 4]);  view_114 = None
        unbind_9 = torch.ops.aten.unbind.int(permute_70);  permute_70 = None
        getitem_107 = unbind_9[0]
        getitem_108 = unbind_9[1]
        getitem_109 = unbind_9[2];  unbind_9 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_107, getitem_108, getitem_109, None, False);  getitem_107 = getitem_108 = getitem_109 = None
        getitem_110 = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
        permute_71 = torch.ops.aten.permute.default(getitem_110, [0, 2, 1, 3]);  getitem_110 = None
        view_115 = torch.ops.aten.view.default(permute_71, [32, 256, 144]);  permute_71 = None
        view_116 = torch.ops.aten.view.default(view_115, [8192, 144]);  view_115 = None
        permute_72 = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_38 = torch.ops.aten.addmm.default(arg92_1, view_116, permute_72);  arg92_1 = view_116 = permute_72 = None
        view_117 = torch.ops.aten.view.default(addmm_38, [32, 256, 144]);  addmm_38 = None
        add_164 = torch.ops.aten.add.Tensor(view_111, view_117);  view_111 = view_117 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_114 = var_mean_22[0]
        getitem_115 = var_mean_22[1];  var_mean_22 = None
        add_165 = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_71 = torch.ops.aten.sub.Tensor(add_164, getitem_115);  getitem_115 = None
        mul_237 = torch.ops.aten.mul.Tensor(sub_71, rsqrt_22);  sub_71 = rsqrt_22 = None
        mul_238 = torch.ops.aten.mul.Tensor(mul_237, arg93_1);  mul_237 = arg93_1 = None
        add_166 = torch.ops.aten.add.Tensor(mul_238, arg94_1);  mul_238 = arg94_1 = None
        view_118 = torch.ops.aten.view.default(add_166, [8192, 144]);  add_166 = None
        permute_73 = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_39 = torch.ops.aten.addmm.default(arg96_1, view_118, permute_73);  arg96_1 = view_118 = permute_73 = None
        view_119 = torch.ops.aten.view.default(addmm_39, [32, 256, 288]);  addmm_39 = None
        sigmoid_46 = torch.ops.aten.sigmoid.default(view_119)
        mul_239 = torch.ops.aten.mul.Tensor(view_119, sigmoid_46);  view_119 = sigmoid_46 = None
        view_120 = torch.ops.aten.view.default(mul_239, [8192, 288]);  mul_239 = None
        permute_74 = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_40 = torch.ops.aten.addmm.default(arg98_1, view_120, permute_74);  arg98_1 = view_120 = permute_74 = None
        view_121 = torch.ops.aten.view.default(addmm_40, [32, 256, 144]);  addmm_40 = None
        add_167 = torch.ops.aten.add.Tensor(add_164, view_121);  add_164 = view_121 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
        getitem_116 = var_mean_23[0]
        getitem_117 = var_mean_23[1];  var_mean_23 = None
        add_168 = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        sub_72 = torch.ops.aten.sub.Tensor(add_167, getitem_117);  getitem_117 = None
        mul_240 = torch.ops.aten.mul.Tensor(sub_72, rsqrt_23);  sub_72 = rsqrt_23 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, arg99_1);  mul_240 = arg99_1 = None
        add_169 = torch.ops.aten.add.Tensor(mul_241, arg100_1);  mul_241 = arg100_1 = None
        view_122 = torch.ops.aten.view.default(add_169, [8192, 144]);  add_169 = None
        permute_75 = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_41 = torch.ops.aten.addmm.default(arg102_1, view_122, permute_75);  arg102_1 = view_122 = permute_75 = None
        view_123 = torch.ops.aten.view.default(addmm_41, [32, 256, 432]);  addmm_41 = None
        view_124 = torch.ops.aten.view.default(view_123, [32, 256, 3, 4, 36]);  view_123 = None
        permute_76 = torch.ops.aten.permute.default(view_124, [2, 0, 3, 1, 4]);  view_124 = None
        unbind_10 = torch.ops.aten.unbind.int(permute_76);  permute_76 = None
        getitem_118 = unbind_10[0]
        getitem_119 = unbind_10[1]
        getitem_120 = unbind_10[2];  unbind_10 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_118, getitem_119, getitem_120, None, False);  getitem_118 = getitem_119 = getitem_120 = None
        getitem_121 = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
        permute_77 = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3]);  getitem_121 = None
        view_125 = torch.ops.aten.view.default(permute_77, [32, 256, 144]);  permute_77 = None
        view_126 = torch.ops.aten.view.default(view_125, [8192, 144]);  view_125 = None
        permute_78 = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_42 = torch.ops.aten.addmm.default(arg104_1, view_126, permute_78);  arg104_1 = view_126 = permute_78 = None
        view_127 = torch.ops.aten.view.default(addmm_42, [32, 256, 144]);  addmm_42 = None
        add_170 = torch.ops.aten.add.Tensor(add_167, view_127);  add_167 = view_127 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_170, [2], correction = 0, keepdim = True)
        getitem_125 = var_mean_24[0]
        getitem_126 = var_mean_24[1];  var_mean_24 = None
        add_171 = torch.ops.aten.add.Tensor(getitem_125, 1e-05);  getitem_125 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_73 = torch.ops.aten.sub.Tensor(add_170, getitem_126);  getitem_126 = None
        mul_242 = torch.ops.aten.mul.Tensor(sub_73, rsqrt_24);  sub_73 = rsqrt_24 = None
        mul_243 = torch.ops.aten.mul.Tensor(mul_242, arg105_1);  mul_242 = arg105_1 = None
        add_172 = torch.ops.aten.add.Tensor(mul_243, arg106_1);  mul_243 = arg106_1 = None
        view_128 = torch.ops.aten.view.default(add_172, [8192, 144]);  add_172 = None
        permute_79 = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_43 = torch.ops.aten.addmm.default(arg108_1, view_128, permute_79);  arg108_1 = view_128 = permute_79 = None
        view_129 = torch.ops.aten.view.default(addmm_43, [32, 256, 288]);  addmm_43 = None
        sigmoid_47 = torch.ops.aten.sigmoid.default(view_129)
        mul_244 = torch.ops.aten.mul.Tensor(view_129, sigmoid_47);  view_129 = sigmoid_47 = None
        view_130 = torch.ops.aten.view.default(mul_244, [8192, 288]);  mul_244 = None
        permute_80 = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_44 = torch.ops.aten.addmm.default(arg110_1, view_130, permute_80);  arg110_1 = view_130 = permute_80 = None
        view_131 = torch.ops.aten.view.default(addmm_44, [32, 256, 144]);  addmm_44 = None
        add_173 = torch.ops.aten.add.Tensor(add_170, view_131);  add_170 = view_131 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_127 = var_mean_25[0]
        getitem_128 = var_mean_25[1];  var_mean_25 = None
        add_174 = torch.ops.aten.add.Tensor(getitem_127, 1e-05);  getitem_127 = None
        rsqrt_25 = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_74 = torch.ops.aten.sub.Tensor(add_173, getitem_128);  add_173 = getitem_128 = None
        mul_245 = torch.ops.aten.mul.Tensor(sub_74, rsqrt_25);  sub_74 = rsqrt_25 = None
        mul_246 = torch.ops.aten.mul.Tensor(mul_245, arg111_1);  mul_245 = arg111_1 = None
        add_175 = torch.ops.aten.add.Tensor(mul_246, arg112_1);  mul_246 = arg112_1 = None
        view_132 = torch.ops.aten.view.default(add_175, [8, 4, 256, -1]);  add_175 = None
        permute_81 = torch.ops.aten.permute.default(view_132, [0, 3, 2, 1]);  view_132 = None
        clone_48 = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_133 = torch.ops.aten.view.default(clone_48, [18432, 16, 2, 2]);  clone_48 = None
        permute_82 = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        clone_49 = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_134 = torch.ops.aten.view.default(clone_49, [8, 144, 32, 32]);  clone_49 = None
        convolution_53 = torch.ops.aten.convolution.default(view_134, arg113_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_134 = arg113_1 = None
        add_176 = torch.ops.aten.add.Tensor(arg115_1, 1e-05);  arg115_1 = None
        sqrt_49 = torch.ops.aten.sqrt.default(add_176);  add_176 = None
        reciprocal_49 = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_247 = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(mul_247, -1);  mul_247 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_75 = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_393);  convolution_53 = unsqueeze_393 = None
        mul_248 = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_395);  sub_75 = unsqueeze_395 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(arg116_1, -1);  arg116_1 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_249 = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_397);  mul_248 = unsqueeze_397 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(arg117_1, -1);  arg117_1 = None
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_177 = torch.ops.aten.add.Tensor(mul_249, unsqueeze_399);  mul_249 = unsqueeze_399 = None
        sigmoid_48 = torch.ops.aten.sigmoid.default(add_177)
        mul_250 = torch.ops.aten.mul.Tensor(add_177, sigmoid_48);  add_177 = sigmoid_48 = None
        cat_3 = torch.ops.aten.cat.default([add_159, mul_250], 1);  add_159 = mul_250 = None
        convolution_54 = torch.ops.aten.convolution.default(cat_3, arg118_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_3 = arg118_1 = None
        add_178 = torch.ops.aten.add.Tensor(arg120_1, 1e-05);  arg120_1 = None
        sqrt_50 = torch.ops.aten.sqrt.default(add_178);  add_178 = None
        reciprocal_50 = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_251 = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_251, -1);  mul_251 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_76 = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_401);  convolution_54 = unsqueeze_401 = None
        mul_252 = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_403);  sub_76 = unsqueeze_403 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(arg121_1, -1);  arg121_1 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_253 = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_405);  mul_252 = unsqueeze_405 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_179 = torch.ops.aten.add.Tensor(mul_253, unsqueeze_407);  mul_253 = unsqueeze_407 = None
        sigmoid_49 = torch.ops.aten.sigmoid.default(add_179)
        mul_254 = torch.ops.aten.mul.Tensor(add_179, sigmoid_49);  add_179 = sigmoid_49 = None
        convolution_55 = torch.ops.aten.convolution.default(mul_254, arg123_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_254 = arg123_1 = None
        add_180 = torch.ops.aten.add.Tensor(arg125_1, 1e-05);  arg125_1 = None
        sqrt_51 = torch.ops.aten.sqrt.default(add_180);  add_180 = None
        reciprocal_51 = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_255 = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_77 = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_409);  convolution_55 = unsqueeze_409 = None
        mul_256 = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_411);  sub_77 = unsqueeze_411 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(arg126_1, -1);  arg126_1 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_413);  mul_256 = unsqueeze_413 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_181 = torch.ops.aten.add.Tensor(mul_257, unsqueeze_415);  mul_257 = unsqueeze_415 = None
        sigmoid_50 = torch.ops.aten.sigmoid.default(add_181)
        mul_258 = torch.ops.aten.mul.Tensor(add_181, sigmoid_50);  add_181 = sigmoid_50 = None
        convolution_56 = torch.ops.aten.convolution.default(mul_258, arg128_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 384);  mul_258 = arg128_1 = None
        add_182 = torch.ops.aten.add.Tensor(arg130_1, 1e-05);  arg130_1 = None
        sqrt_52 = torch.ops.aten.sqrt.default(add_182);  add_182 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_259 = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(mul_259, -1);  mul_259 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_78 = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_417);  convolution_56 = unsqueeze_417 = None
        mul_260 = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_419);  sub_78 = unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(arg131_1, -1);  arg131_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_261 = torch.ops.aten.mul.Tensor(mul_260, unsqueeze_421);  mul_260 = unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_183 = torch.ops.aten.add.Tensor(mul_261, unsqueeze_423);  mul_261 = unsqueeze_423 = None
        sigmoid_51 = torch.ops.aten.sigmoid.default(add_183)
        mul_262 = torch.ops.aten.mul.Tensor(add_183, sigmoid_51);  add_183 = sigmoid_51 = None
        convolution_57 = torch.ops.aten.convolution.default(mul_262, arg133_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_262 = arg133_1 = None
        add_184 = torch.ops.aten.add.Tensor(arg135_1, 1e-05);  arg135_1 = None
        sqrt_53 = torch.ops.aten.sqrt.default(add_184);  add_184 = None
        reciprocal_53 = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_263 = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_263, -1);  mul_263 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_79 = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_425);  convolution_57 = unsqueeze_425 = None
        mul_264 = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_427);  sub_79 = unsqueeze_427 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(arg136_1, -1);  arg136_1 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_265 = torch.ops.aten.mul.Tensor(mul_264, unsqueeze_429);  mul_264 = unsqueeze_429 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(arg137_1, -1);  arg137_1 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_185 = torch.ops.aten.add.Tensor(mul_265, unsqueeze_431);  mul_265 = unsqueeze_431 = None
        convolution_58 = torch.ops.aten.convolution.default(add_185, arg138_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg138_1 = None
        add_186 = torch.ops.aten.add.Tensor(arg140_1, 1e-05);  arg140_1 = None
        sqrt_54 = torch.ops.aten.sqrt.default(add_186);  add_186 = None
        reciprocal_54 = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_266 = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(mul_266, -1);  mul_266 = None
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_80 = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_433);  convolution_58 = unsqueeze_433 = None
        mul_267 = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_435);  sub_80 = unsqueeze_435 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_268 = torch.ops.aten.mul.Tensor(mul_267, unsqueeze_437);  mul_267 = unsqueeze_437 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(arg142_1, -1);  arg142_1 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_187 = torch.ops.aten.add.Tensor(mul_268, unsqueeze_439);  mul_268 = unsqueeze_439 = None
        sigmoid_52 = torch.ops.aten.sigmoid.default(add_187)
        mul_269 = torch.ops.aten.mul.Tensor(add_187, sigmoid_52);  add_187 = sigmoid_52 = None
        convolution_59 = torch.ops.aten.convolution.default(mul_269, arg143_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_269 = arg143_1 = None
        view_135 = torch.ops.aten.view.default(convolution_59, [12288, 2, 8, 2]);  convolution_59 = None
        permute_83 = torch.ops.aten.permute.default(view_135, [0, 2, 1, 3]);  view_135 = None
        clone_50 = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_136 = torch.ops.aten.view.default(clone_50, [8, 192, 64, 4]);  clone_50 = None
        permute_84 = torch.ops.aten.permute.default(view_136, [0, 3, 2, 1]);  view_136 = None
        clone_51 = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_137 = torch.ops.aten.view.default(clone_51, [32, 64, 192]);  clone_51 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(view_137, [2], correction = 0, keepdim = True)
        getitem_129 = var_mean_26[0]
        getitem_130 = var_mean_26[1];  var_mean_26 = None
        add_188 = torch.ops.aten.add.Tensor(getitem_129, 1e-05);  getitem_129 = None
        rsqrt_26 = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        sub_81 = torch.ops.aten.sub.Tensor(view_137, getitem_130);  getitem_130 = None
        mul_270 = torch.ops.aten.mul.Tensor(sub_81, rsqrt_26);  sub_81 = rsqrt_26 = None
        mul_271 = torch.ops.aten.mul.Tensor(mul_270, arg144_1);  mul_270 = arg144_1 = None
        add_189 = torch.ops.aten.add.Tensor(mul_271, arg145_1);  mul_271 = arg145_1 = None
        view_138 = torch.ops.aten.view.default(add_189, [2048, 192]);  add_189 = None
        permute_85 = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_45 = torch.ops.aten.addmm.default(arg147_1, view_138, permute_85);  arg147_1 = view_138 = permute_85 = None
        view_139 = torch.ops.aten.view.default(addmm_45, [32, 64, 576]);  addmm_45 = None
        view_140 = torch.ops.aten.view.default(view_139, [32, 64, 3, 4, 48]);  view_139 = None
        permute_86 = torch.ops.aten.permute.default(view_140, [2, 0, 3, 1, 4]);  view_140 = None
        unbind_11 = torch.ops.aten.unbind.int(permute_86);  permute_86 = None
        getitem_131 = unbind_11[0]
        getitem_132 = unbind_11[1]
        getitem_133 = unbind_11[2];  unbind_11 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_131, getitem_132, getitem_133, None, False);  getitem_131 = getitem_132 = getitem_133 = None
        getitem_134 = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
        permute_87 = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        view_141 = torch.ops.aten.view.default(permute_87, [32, 64, 192]);  permute_87 = None
        view_142 = torch.ops.aten.view.default(view_141, [2048, 192]);  view_141 = None
        permute_88 = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_46 = torch.ops.aten.addmm.default(arg149_1, view_142, permute_88);  arg149_1 = view_142 = permute_88 = None
        view_143 = torch.ops.aten.view.default(addmm_46, [32, 64, 192]);  addmm_46 = None
        add_190 = torch.ops.aten.add.Tensor(view_137, view_143);  view_137 = view_143 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_190, [2], correction = 0, keepdim = True)
        getitem_138 = var_mean_27[0]
        getitem_139 = var_mean_27[1];  var_mean_27 = None
        add_191 = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
        rsqrt_27 = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_82 = torch.ops.aten.sub.Tensor(add_190, getitem_139);  getitem_139 = None
        mul_272 = torch.ops.aten.mul.Tensor(sub_82, rsqrt_27);  sub_82 = rsqrt_27 = None
        mul_273 = torch.ops.aten.mul.Tensor(mul_272, arg150_1);  mul_272 = arg150_1 = None
        add_192 = torch.ops.aten.add.Tensor(mul_273, arg151_1);  mul_273 = arg151_1 = None
        view_144 = torch.ops.aten.view.default(add_192, [2048, 192]);  add_192 = None
        permute_89 = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
        addmm_47 = torch.ops.aten.addmm.default(arg153_1, view_144, permute_89);  arg153_1 = view_144 = permute_89 = None
        view_145 = torch.ops.aten.view.default(addmm_47, [32, 64, 384]);  addmm_47 = None
        sigmoid_53 = torch.ops.aten.sigmoid.default(view_145)
        mul_274 = torch.ops.aten.mul.Tensor(view_145, sigmoid_53);  view_145 = sigmoid_53 = None
        view_146 = torch.ops.aten.view.default(mul_274, [2048, 384]);  mul_274 = None
        permute_90 = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
        addmm_48 = torch.ops.aten.addmm.default(arg155_1, view_146, permute_90);  arg155_1 = view_146 = permute_90 = None
        view_147 = torch.ops.aten.view.default(addmm_48, [32, 64, 192]);  addmm_48 = None
        add_193 = torch.ops.aten.add.Tensor(add_190, view_147);  add_190 = view_147 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
        getitem_140 = var_mean_28[0]
        getitem_141 = var_mean_28[1];  var_mean_28 = None
        add_194 = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
        rsqrt_28 = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        sub_83 = torch.ops.aten.sub.Tensor(add_193, getitem_141);  getitem_141 = None
        mul_275 = torch.ops.aten.mul.Tensor(sub_83, rsqrt_28);  sub_83 = rsqrt_28 = None
        mul_276 = torch.ops.aten.mul.Tensor(mul_275, arg156_1);  mul_275 = arg156_1 = None
        add_195 = torch.ops.aten.add.Tensor(mul_276, arg157_1);  mul_276 = arg157_1 = None
        view_148 = torch.ops.aten.view.default(add_195, [2048, 192]);  add_195 = None
        permute_91 = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
        addmm_49 = torch.ops.aten.addmm.default(arg159_1, view_148, permute_91);  arg159_1 = view_148 = permute_91 = None
        view_149 = torch.ops.aten.view.default(addmm_49, [32, 64, 576]);  addmm_49 = None
        view_150 = torch.ops.aten.view.default(view_149, [32, 64, 3, 4, 48]);  view_149 = None
        permute_92 = torch.ops.aten.permute.default(view_150, [2, 0, 3, 1, 4]);  view_150 = None
        unbind_12 = torch.ops.aten.unbind.int(permute_92);  permute_92 = None
        getitem_142 = unbind_12[0]
        getitem_143 = unbind_12[1]
        getitem_144 = unbind_12[2];  unbind_12 = None
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_142, getitem_143, getitem_144, None, False);  getitem_142 = getitem_143 = getitem_144 = None
        getitem_145 = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None
        permute_93 = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None
        view_151 = torch.ops.aten.view.default(permute_93, [32, 64, 192]);  permute_93 = None
        view_152 = torch.ops.aten.view.default(view_151, [2048, 192]);  view_151 = None
        permute_94 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        addmm_50 = torch.ops.aten.addmm.default(arg161_1, view_152, permute_94);  arg161_1 = view_152 = permute_94 = None
        view_153 = torch.ops.aten.view.default(addmm_50, [32, 64, 192]);  addmm_50 = None
        add_196 = torch.ops.aten.add.Tensor(add_193, view_153);  add_193 = view_153 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_196, [2], correction = 0, keepdim = True)
        getitem_149 = var_mean_29[0]
        getitem_150 = var_mean_29[1];  var_mean_29 = None
        add_197 = torch.ops.aten.add.Tensor(getitem_149, 1e-05);  getitem_149 = None
        rsqrt_29 = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_84 = torch.ops.aten.sub.Tensor(add_196, getitem_150);  getitem_150 = None
        mul_277 = torch.ops.aten.mul.Tensor(sub_84, rsqrt_29);  sub_84 = rsqrt_29 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_277, arg162_1);  mul_277 = arg162_1 = None
        add_198 = torch.ops.aten.add.Tensor(mul_278, arg163_1);  mul_278 = arg163_1 = None
        view_154 = torch.ops.aten.view.default(add_198, [2048, 192]);  add_198 = None
        permute_95 = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
        addmm_51 = torch.ops.aten.addmm.default(arg165_1, view_154, permute_95);  arg165_1 = view_154 = permute_95 = None
        view_155 = torch.ops.aten.view.default(addmm_51, [32, 64, 384]);  addmm_51 = None
        sigmoid_54 = torch.ops.aten.sigmoid.default(view_155)
        mul_279 = torch.ops.aten.mul.Tensor(view_155, sigmoid_54);  view_155 = sigmoid_54 = None
        view_156 = torch.ops.aten.view.default(mul_279, [2048, 384]);  mul_279 = None
        permute_96 = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
        addmm_52 = torch.ops.aten.addmm.default(arg167_1, view_156, permute_96);  arg167_1 = view_156 = permute_96 = None
        view_157 = torch.ops.aten.view.default(addmm_52, [32, 64, 192]);  addmm_52 = None
        add_199 = torch.ops.aten.add.Tensor(add_196, view_157);  add_196 = view_157 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_199, [2], correction = 0, keepdim = True)
        getitem_151 = var_mean_30[0]
        getitem_152 = var_mean_30[1];  var_mean_30 = None
        add_200 = torch.ops.aten.add.Tensor(getitem_151, 1e-05);  getitem_151 = None
        rsqrt_30 = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        sub_85 = torch.ops.aten.sub.Tensor(add_199, getitem_152);  getitem_152 = None
        mul_280 = torch.ops.aten.mul.Tensor(sub_85, rsqrt_30);  sub_85 = rsqrt_30 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, arg168_1);  mul_280 = arg168_1 = None
        add_201 = torch.ops.aten.add.Tensor(mul_281, arg169_1);  mul_281 = arg169_1 = None
        view_158 = torch.ops.aten.view.default(add_201, [2048, 192]);  add_201 = None
        permute_97 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_53 = torch.ops.aten.addmm.default(arg171_1, view_158, permute_97);  arg171_1 = view_158 = permute_97 = None
        view_159 = torch.ops.aten.view.default(addmm_53, [32, 64, 576]);  addmm_53 = None
        view_160 = torch.ops.aten.view.default(view_159, [32, 64, 3, 4, 48]);  view_159 = None
        permute_98 = torch.ops.aten.permute.default(view_160, [2, 0, 3, 1, 4]);  view_160 = None
        unbind_13 = torch.ops.aten.unbind.int(permute_98);  permute_98 = None
        getitem_153 = unbind_13[0]
        getitem_154 = unbind_13[1]
        getitem_155 = unbind_13[2];  unbind_13 = None
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_153, getitem_154, getitem_155, None, False);  getitem_153 = getitem_154 = getitem_155 = None
        getitem_156 = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None
        permute_99 = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3]);  getitem_156 = None
        view_161 = torch.ops.aten.view.default(permute_99, [32, 64, 192]);  permute_99 = None
        view_162 = torch.ops.aten.view.default(view_161, [2048, 192]);  view_161 = None
        permute_100 = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
        addmm_54 = torch.ops.aten.addmm.default(arg173_1, view_162, permute_100);  arg173_1 = view_162 = permute_100 = None
        view_163 = torch.ops.aten.view.default(addmm_54, [32, 64, 192]);  addmm_54 = None
        add_202 = torch.ops.aten.add.Tensor(add_199, view_163);  add_199 = view_163 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
        getitem_160 = var_mean_31[0]
        getitem_161 = var_mean_31[1];  var_mean_31 = None
        add_203 = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_31 = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_86 = torch.ops.aten.sub.Tensor(add_202, getitem_161);  getitem_161 = None
        mul_282 = torch.ops.aten.mul.Tensor(sub_86, rsqrt_31);  sub_86 = rsqrt_31 = None
        mul_283 = torch.ops.aten.mul.Tensor(mul_282, arg174_1);  mul_282 = arg174_1 = None
        add_204 = torch.ops.aten.add.Tensor(mul_283, arg175_1);  mul_283 = arg175_1 = None
        view_164 = torch.ops.aten.view.default(add_204, [2048, 192]);  add_204 = None
        permute_101 = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_55 = torch.ops.aten.addmm.default(arg177_1, view_164, permute_101);  arg177_1 = view_164 = permute_101 = None
        view_165 = torch.ops.aten.view.default(addmm_55, [32, 64, 384]);  addmm_55 = None
        sigmoid_55 = torch.ops.aten.sigmoid.default(view_165)
        mul_284 = torch.ops.aten.mul.Tensor(view_165, sigmoid_55);  view_165 = sigmoid_55 = None
        view_166 = torch.ops.aten.view.default(mul_284, [2048, 384]);  mul_284 = None
        permute_102 = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_56 = torch.ops.aten.addmm.default(arg179_1, view_166, permute_102);  arg179_1 = view_166 = permute_102 = None
        view_167 = torch.ops.aten.view.default(addmm_56, [32, 64, 192]);  addmm_56 = None
        add_205 = torch.ops.aten.add.Tensor(add_202, view_167);  add_202 = view_167 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_205, [2], correction = 0, keepdim = True)
        getitem_162 = var_mean_32[0]
        getitem_163 = var_mean_32[1];  var_mean_32 = None
        add_206 = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
        rsqrt_32 = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        sub_87 = torch.ops.aten.sub.Tensor(add_205, getitem_163);  getitem_163 = None
        mul_285 = torch.ops.aten.mul.Tensor(sub_87, rsqrt_32);  sub_87 = rsqrt_32 = None
        mul_286 = torch.ops.aten.mul.Tensor(mul_285, arg180_1);  mul_285 = arg180_1 = None
        add_207 = torch.ops.aten.add.Tensor(mul_286, arg181_1);  mul_286 = arg181_1 = None
        view_168 = torch.ops.aten.view.default(add_207, [2048, 192]);  add_207 = None
        permute_103 = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
        addmm_57 = torch.ops.aten.addmm.default(arg183_1, view_168, permute_103);  arg183_1 = view_168 = permute_103 = None
        view_169 = torch.ops.aten.view.default(addmm_57, [32, 64, 576]);  addmm_57 = None
        view_170 = torch.ops.aten.view.default(view_169, [32, 64, 3, 4, 48]);  view_169 = None
        permute_104 = torch.ops.aten.permute.default(view_170, [2, 0, 3, 1, 4]);  view_170 = None
        unbind_14 = torch.ops.aten.unbind.int(permute_104);  permute_104 = None
        getitem_164 = unbind_14[0]
        getitem_165 = unbind_14[1]
        getitem_166 = unbind_14[2];  unbind_14 = None
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_164, getitem_165, getitem_166, None, False);  getitem_164 = getitem_165 = getitem_166 = None
        getitem_167 = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None
        permute_105 = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
        view_171 = torch.ops.aten.view.default(permute_105, [32, 64, 192]);  permute_105 = None
        view_172 = torch.ops.aten.view.default(view_171, [2048, 192]);  view_171 = None
        permute_106 = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
        addmm_58 = torch.ops.aten.addmm.default(arg185_1, view_172, permute_106);  arg185_1 = view_172 = permute_106 = None
        view_173 = torch.ops.aten.view.default(addmm_58, [32, 64, 192]);  addmm_58 = None
        add_208 = torch.ops.aten.add.Tensor(add_205, view_173);  add_205 = view_173 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_208, [2], correction = 0, keepdim = True)
        getitem_171 = var_mean_33[0]
        getitem_172 = var_mean_33[1];  var_mean_33 = None
        add_209 = torch.ops.aten.add.Tensor(getitem_171, 1e-05);  getitem_171 = None
        rsqrt_33 = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
        sub_88 = torch.ops.aten.sub.Tensor(add_208, getitem_172);  getitem_172 = None
        mul_287 = torch.ops.aten.mul.Tensor(sub_88, rsqrt_33);  sub_88 = rsqrt_33 = None
        mul_288 = torch.ops.aten.mul.Tensor(mul_287, arg186_1);  mul_287 = arg186_1 = None
        add_210 = torch.ops.aten.add.Tensor(mul_288, arg187_1);  mul_288 = arg187_1 = None
        view_174 = torch.ops.aten.view.default(add_210, [2048, 192]);  add_210 = None
        permute_107 = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
        addmm_59 = torch.ops.aten.addmm.default(arg189_1, view_174, permute_107);  arg189_1 = view_174 = permute_107 = None
        view_175 = torch.ops.aten.view.default(addmm_59, [32, 64, 384]);  addmm_59 = None
        sigmoid_56 = torch.ops.aten.sigmoid.default(view_175)
        mul_289 = torch.ops.aten.mul.Tensor(view_175, sigmoid_56);  view_175 = sigmoid_56 = None
        view_176 = torch.ops.aten.view.default(mul_289, [2048, 384]);  mul_289 = None
        permute_108 = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
        addmm_60 = torch.ops.aten.addmm.default(arg191_1, view_176, permute_108);  arg191_1 = view_176 = permute_108 = None
        view_177 = torch.ops.aten.view.default(addmm_60, [32, 64, 192]);  addmm_60 = None
        add_211 = torch.ops.aten.add.Tensor(add_208, view_177);  add_208 = view_177 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_211, [2], correction = 0, keepdim = True)
        getitem_173 = var_mean_34[0]
        getitem_174 = var_mean_34[1];  var_mean_34 = None
        add_212 = torch.ops.aten.add.Tensor(getitem_173, 1e-05);  getitem_173 = None
        rsqrt_34 = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
        sub_89 = torch.ops.aten.sub.Tensor(add_211, getitem_174);  add_211 = getitem_174 = None
        mul_290 = torch.ops.aten.mul.Tensor(sub_89, rsqrt_34);  sub_89 = rsqrt_34 = None
        mul_291 = torch.ops.aten.mul.Tensor(mul_290, arg192_1);  mul_290 = arg192_1 = None
        add_213 = torch.ops.aten.add.Tensor(mul_291, arg193_1);  mul_291 = arg193_1 = None
        view_178 = torch.ops.aten.view.default(add_213, [8, 4, 64, -1]);  add_213 = None
        permute_109 = torch.ops.aten.permute.default(view_178, [0, 3, 2, 1]);  view_178 = None
        clone_64 = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
        view_179 = torch.ops.aten.view.default(clone_64, [12288, 8, 2, 2]);  clone_64 = None
        permute_110 = torch.ops.aten.permute.default(view_179, [0, 2, 1, 3]);  view_179 = None
        clone_65 = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_180 = torch.ops.aten.view.default(clone_65, [8, 192, 16, 16]);  clone_65 = None
        convolution_60 = torch.ops.aten.convolution.default(view_180, arg194_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_180 = arg194_1 = None
        add_214 = torch.ops.aten.add.Tensor(arg196_1, 1e-05);  arg196_1 = None
        sqrt_55 = torch.ops.aten.sqrt.default(add_214);  add_214 = None
        reciprocal_55 = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_292 = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(mul_292, -1);  mul_292 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_90 = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_441);  convolution_60 = unsqueeze_441 = None
        mul_293 = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_443);  sub_90 = unsqueeze_443 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(arg197_1, -1);  arg197_1 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_294 = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_445);  mul_293 = unsqueeze_445 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(arg198_1, -1);  arg198_1 = None
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_215 = torch.ops.aten.add.Tensor(mul_294, unsqueeze_447);  mul_294 = unsqueeze_447 = None
        sigmoid_57 = torch.ops.aten.sigmoid.default(add_215)
        mul_295 = torch.ops.aten.mul.Tensor(add_215, sigmoid_57);  add_215 = sigmoid_57 = None
        cat_4 = torch.ops.aten.cat.default([add_185, mul_295], 1);  add_185 = mul_295 = None
        convolution_61 = torch.ops.aten.convolution.default(cat_4, arg199_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_4 = arg199_1 = None
        add_216 = torch.ops.aten.add.Tensor(arg201_1, 1e-05);  arg201_1 = None
        sqrt_56 = torch.ops.aten.sqrt.default(add_216);  add_216 = None
        reciprocal_56 = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_296 = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448 = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(mul_296, -1);  mul_296 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_91 = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_449);  convolution_61 = unsqueeze_449 = None
        mul_297 = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_451);  sub_91 = unsqueeze_451 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(arg202_1, -1);  arg202_1 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_298 = torch.ops.aten.mul.Tensor(mul_297, unsqueeze_453);  mul_297 = unsqueeze_453 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(arg203_1, -1);  arg203_1 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_217 = torch.ops.aten.add.Tensor(mul_298, unsqueeze_455);  mul_298 = unsqueeze_455 = None
        sigmoid_58 = torch.ops.aten.sigmoid.default(add_217)
        mul_299 = torch.ops.aten.mul.Tensor(add_217, sigmoid_58);  add_217 = sigmoid_58 = None
        convolution_62 = torch.ops.aten.convolution.default(mul_299, arg204_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_299 = arg204_1 = None
        add_218 = torch.ops.aten.add.Tensor(arg206_1, 1e-05);  arg206_1 = None
        sqrt_57 = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_57 = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_300 = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458 = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_459 = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_92 = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_457);  convolution_62 = unsqueeze_457 = None
        mul_301 = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_459);  sub_92 = unsqueeze_459 = None
        unsqueeze_460 = torch.ops.aten.unsqueeze.default(arg207_1, -1);  arg207_1 = None
        unsqueeze_461 = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_461);  mul_301 = unsqueeze_461 = None
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(arg208_1, -1);  arg208_1 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_219 = torch.ops.aten.add.Tensor(mul_302, unsqueeze_463);  mul_302 = unsqueeze_463 = None
        sigmoid_59 = torch.ops.aten.sigmoid.default(add_219)
        mul_303 = torch.ops.aten.mul.Tensor(add_219, sigmoid_59);  add_219 = sigmoid_59 = None
        convolution_63 = torch.ops.aten.convolution.default(mul_303, arg209_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 512);  mul_303 = arg209_1 = None
        add_220 = torch.ops.aten.add.Tensor(arg211_1, 1e-05);  arg211_1 = None
        sqrt_58 = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_58 = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_304 = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(mul_304, -1);  mul_304 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_93 = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_465);  convolution_63 = unsqueeze_465 = None
        mul_305 = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_467);  sub_93 = unsqueeze_467 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(arg212_1, -1);  arg212_1 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_306 = torch.ops.aten.mul.Tensor(mul_305, unsqueeze_469);  mul_305 = unsqueeze_469 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(arg213_1, -1);  arg213_1 = None
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_221 = torch.ops.aten.add.Tensor(mul_306, unsqueeze_471);  mul_306 = unsqueeze_471 = None
        sigmoid_60 = torch.ops.aten.sigmoid.default(add_221)
        mul_307 = torch.ops.aten.mul.Tensor(add_221, sigmoid_60);  add_221 = sigmoid_60 = None
        convolution_64 = torch.ops.aten.convolution.default(mul_307, arg214_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_307 = arg214_1 = None
        add_222 = torch.ops.aten.add.Tensor(arg216_1, 1e-05);  arg216_1 = None
        sqrt_59 = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_59 = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_308 = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(mul_308, -1);  mul_308 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_94 = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_473);  convolution_64 = unsqueeze_473 = None
        mul_309 = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_475);  sub_94 = unsqueeze_475 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(arg217_1, -1);  arg217_1 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_310 = torch.ops.aten.mul.Tensor(mul_309, unsqueeze_477);  mul_309 = unsqueeze_477 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(arg218_1, -1);  arg218_1 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_223 = torch.ops.aten.add.Tensor(mul_310, unsqueeze_479);  mul_310 = unsqueeze_479 = None
        convolution_65 = torch.ops.aten.convolution.default(add_223, arg219_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  arg219_1 = None
        add_224 = torch.ops.aten.add.Tensor(arg221_1, 1e-05);  arg221_1 = None
        sqrt_60 = torch.ops.aten.sqrt.default(add_224);  add_224 = None
        reciprocal_60 = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_311 = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(mul_311, -1);  mul_311 = None
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_95 = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_481);  convolution_65 = unsqueeze_481 = None
        mul_312 = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_483);  sub_95 = unsqueeze_483 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(arg222_1, -1);  arg222_1 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_313 = torch.ops.aten.mul.Tensor(mul_312, unsqueeze_485);  mul_312 = unsqueeze_485 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(arg223_1, -1);  arg223_1 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_225 = torch.ops.aten.add.Tensor(mul_313, unsqueeze_487);  mul_313 = unsqueeze_487 = None
        sigmoid_61 = torch.ops.aten.sigmoid.default(add_225)
        mul_314 = torch.ops.aten.mul.Tensor(add_225, sigmoid_61);  add_225 = sigmoid_61 = None
        convolution_66 = torch.ops.aten.convolution.default(mul_314, arg224_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_314 = arg224_1 = None
        view_181 = torch.ops.aten.view.default(convolution_66, [7680, 2, 4, 2]);  convolution_66 = None
        permute_111 = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        clone_66 = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_182 = torch.ops.aten.view.default(clone_66, [8, 240, 16, 4]);  clone_66 = None
        permute_112 = torch.ops.aten.permute.default(view_182, [0, 3, 2, 1]);  view_182 = None
        clone_67 = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_183 = torch.ops.aten.view.default(clone_67, [32, 16, 240]);  clone_67 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(view_183, [2], correction = 0, keepdim = True)
        getitem_175 = var_mean_35[0]
        getitem_176 = var_mean_35[1];  var_mean_35 = None
        add_226 = torch.ops.aten.add.Tensor(getitem_175, 1e-05);  getitem_175 = None
        rsqrt_35 = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_96 = torch.ops.aten.sub.Tensor(view_183, getitem_176);  getitem_176 = None
        mul_315 = torch.ops.aten.mul.Tensor(sub_96, rsqrt_35);  sub_96 = rsqrt_35 = None
        mul_316 = torch.ops.aten.mul.Tensor(mul_315, arg225_1);  mul_315 = arg225_1 = None
        add_227 = torch.ops.aten.add.Tensor(mul_316, arg226_1);  mul_316 = arg226_1 = None
        view_184 = torch.ops.aten.view.default(add_227, [512, 240]);  add_227 = None
        permute_113 = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_61 = torch.ops.aten.addmm.default(arg228_1, view_184, permute_113);  arg228_1 = view_184 = permute_113 = None
        view_185 = torch.ops.aten.view.default(addmm_61, [32, 16, 720]);  addmm_61 = None
        view_186 = torch.ops.aten.view.default(view_185, [32, 16, 3, 4, 60]);  view_185 = None
        permute_114 = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
        unbind_15 = torch.ops.aten.unbind.int(permute_114);  permute_114 = None
        getitem_177 = unbind_15[0]
        getitem_178 = unbind_15[1]
        getitem_179 = unbind_15[2];  unbind_15 = None
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_177, getitem_178, getitem_179, None, False);  getitem_177 = getitem_178 = getitem_179 = None
        getitem_180 = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None
        permute_115 = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
        view_187 = torch.ops.aten.view.default(permute_115, [32, 16, 240]);  permute_115 = None
        view_188 = torch.ops.aten.view.default(view_187, [512, 240]);  view_187 = None
        permute_116 = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_62 = torch.ops.aten.addmm.default(arg230_1, view_188, permute_116);  arg230_1 = view_188 = permute_116 = None
        view_189 = torch.ops.aten.view.default(addmm_62, [32, 16, 240]);  addmm_62 = None
        add_228 = torch.ops.aten.add.Tensor(view_183, view_189);  view_183 = view_189 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_228, [2], correction = 0, keepdim = True)
        getitem_184 = var_mean_36[0]
        getitem_185 = var_mean_36[1];  var_mean_36 = None
        add_229 = torch.ops.aten.add.Tensor(getitem_184, 1e-05);  getitem_184 = None
        rsqrt_36 = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_97 = torch.ops.aten.sub.Tensor(add_228, getitem_185);  getitem_185 = None
        mul_317 = torch.ops.aten.mul.Tensor(sub_97, rsqrt_36);  sub_97 = rsqrt_36 = None
        mul_318 = torch.ops.aten.mul.Tensor(mul_317, arg231_1);  mul_317 = arg231_1 = None
        add_230 = torch.ops.aten.add.Tensor(mul_318, arg232_1);  mul_318 = arg232_1 = None
        view_190 = torch.ops.aten.view.default(add_230, [512, 240]);  add_230 = None
        permute_117 = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_63 = torch.ops.aten.addmm.default(arg234_1, view_190, permute_117);  arg234_1 = view_190 = permute_117 = None
        view_191 = torch.ops.aten.view.default(addmm_63, [32, 16, 480]);  addmm_63 = None
        sigmoid_62 = torch.ops.aten.sigmoid.default(view_191)
        mul_319 = torch.ops.aten.mul.Tensor(view_191, sigmoid_62);  view_191 = sigmoid_62 = None
        view_192 = torch.ops.aten.view.default(mul_319, [512, 480]);  mul_319 = None
        permute_118 = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_64 = torch.ops.aten.addmm.default(arg236_1, view_192, permute_118);  arg236_1 = view_192 = permute_118 = None
        view_193 = torch.ops.aten.view.default(addmm_64, [32, 16, 240]);  addmm_64 = None
        add_231 = torch.ops.aten.add.Tensor(add_228, view_193);  add_228 = view_193 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_231, [2], correction = 0, keepdim = True)
        getitem_186 = var_mean_37[0]
        getitem_187 = var_mean_37[1];  var_mean_37 = None
        add_232 = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_37 = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        sub_98 = torch.ops.aten.sub.Tensor(add_231, getitem_187);  getitem_187 = None
        mul_320 = torch.ops.aten.mul.Tensor(sub_98, rsqrt_37);  sub_98 = rsqrt_37 = None
        mul_321 = torch.ops.aten.mul.Tensor(mul_320, arg237_1);  mul_320 = arg237_1 = None
        add_233 = torch.ops.aten.add.Tensor(mul_321, arg238_1);  mul_321 = arg238_1 = None
        view_194 = torch.ops.aten.view.default(add_233, [512, 240]);  add_233 = None
        permute_119 = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_65 = torch.ops.aten.addmm.default(arg240_1, view_194, permute_119);  arg240_1 = view_194 = permute_119 = None
        view_195 = torch.ops.aten.view.default(addmm_65, [32, 16, 720]);  addmm_65 = None
        view_196 = torch.ops.aten.view.default(view_195, [32, 16, 3, 4, 60]);  view_195 = None
        permute_120 = torch.ops.aten.permute.default(view_196, [2, 0, 3, 1, 4]);  view_196 = None
        unbind_16 = torch.ops.aten.unbind.int(permute_120);  permute_120 = None
        getitem_188 = unbind_16[0]
        getitem_189 = unbind_16[1]
        getitem_190 = unbind_16[2];  unbind_16 = None
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_188, getitem_189, getitem_190, None, False);  getitem_188 = getitem_189 = getitem_190 = None
        getitem_191 = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None
        permute_121 = torch.ops.aten.permute.default(getitem_191, [0, 2, 1, 3]);  getitem_191 = None
        view_197 = torch.ops.aten.view.default(permute_121, [32, 16, 240]);  permute_121 = None
        view_198 = torch.ops.aten.view.default(view_197, [512, 240]);  view_197 = None
        permute_122 = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_66 = torch.ops.aten.addmm.default(arg242_1, view_198, permute_122);  arg242_1 = view_198 = permute_122 = None
        view_199 = torch.ops.aten.view.default(addmm_66, [32, 16, 240]);  addmm_66 = None
        add_234 = torch.ops.aten.add.Tensor(add_231, view_199);  add_231 = view_199 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_234, [2], correction = 0, keepdim = True)
        getitem_195 = var_mean_38[0]
        getitem_196 = var_mean_38[1];  var_mean_38 = None
        add_235 = torch.ops.aten.add.Tensor(getitem_195, 1e-05);  getitem_195 = None
        rsqrt_38 = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
        sub_99 = torch.ops.aten.sub.Tensor(add_234, getitem_196);  getitem_196 = None
        mul_322 = torch.ops.aten.mul.Tensor(sub_99, rsqrt_38);  sub_99 = rsqrt_38 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_322, arg243_1);  mul_322 = arg243_1 = None
        add_236 = torch.ops.aten.add.Tensor(mul_323, arg244_1);  mul_323 = arg244_1 = None
        view_200 = torch.ops.aten.view.default(add_236, [512, 240]);  add_236 = None
        permute_123 = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_67 = torch.ops.aten.addmm.default(arg246_1, view_200, permute_123);  arg246_1 = view_200 = permute_123 = None
        view_201 = torch.ops.aten.view.default(addmm_67, [32, 16, 480]);  addmm_67 = None
        sigmoid_63 = torch.ops.aten.sigmoid.default(view_201)
        mul_324 = torch.ops.aten.mul.Tensor(view_201, sigmoid_63);  view_201 = sigmoid_63 = None
        view_202 = torch.ops.aten.view.default(mul_324, [512, 480]);  mul_324 = None
        permute_124 = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_68 = torch.ops.aten.addmm.default(arg248_1, view_202, permute_124);  arg248_1 = view_202 = permute_124 = None
        view_203 = torch.ops.aten.view.default(addmm_68, [32, 16, 240]);  addmm_68 = None
        add_237 = torch.ops.aten.add.Tensor(add_234, view_203);  add_234 = view_203 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_237, [2], correction = 0, keepdim = True)
        getitem_197 = var_mean_39[0]
        getitem_198 = var_mean_39[1];  var_mean_39 = None
        add_238 = torch.ops.aten.add.Tensor(getitem_197, 1e-05);  getitem_197 = None
        rsqrt_39 = torch.ops.aten.rsqrt.default(add_238);  add_238 = None
        sub_100 = torch.ops.aten.sub.Tensor(add_237, getitem_198);  getitem_198 = None
        mul_325 = torch.ops.aten.mul.Tensor(sub_100, rsqrt_39);  sub_100 = rsqrt_39 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, arg249_1);  mul_325 = arg249_1 = None
        add_239 = torch.ops.aten.add.Tensor(mul_326, arg250_1);  mul_326 = arg250_1 = None
        view_204 = torch.ops.aten.view.default(add_239, [512, 240]);  add_239 = None
        permute_125 = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_69 = torch.ops.aten.addmm.default(arg252_1, view_204, permute_125);  arg252_1 = view_204 = permute_125 = None
        view_205 = torch.ops.aten.view.default(addmm_69, [32, 16, 720]);  addmm_69 = None
        view_206 = torch.ops.aten.view.default(view_205, [32, 16, 3, 4, 60]);  view_205 = None
        permute_126 = torch.ops.aten.permute.default(view_206, [2, 0, 3, 1, 4]);  view_206 = None
        unbind_17 = torch.ops.aten.unbind.int(permute_126);  permute_126 = None
        getitem_199 = unbind_17[0]
        getitem_200 = unbind_17[1]
        getitem_201 = unbind_17[2];  unbind_17 = None
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_199, getitem_200, getitem_201, None, False);  getitem_199 = getitem_200 = getitem_201 = None
        getitem_202 = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None
        permute_127 = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None
        view_207 = torch.ops.aten.view.default(permute_127, [32, 16, 240]);  permute_127 = None
        view_208 = torch.ops.aten.view.default(view_207, [512, 240]);  view_207 = None
        permute_128 = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_70 = torch.ops.aten.addmm.default(arg254_1, view_208, permute_128);  arg254_1 = view_208 = permute_128 = None
        view_209 = torch.ops.aten.view.default(addmm_70, [32, 16, 240]);  addmm_70 = None
        add_240 = torch.ops.aten.add.Tensor(add_237, view_209);  add_237 = view_209 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_240, [2], correction = 0, keepdim = True)
        getitem_206 = var_mean_40[0]
        getitem_207 = var_mean_40[1];  var_mean_40 = None
        add_241 = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_40 = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
        sub_101 = torch.ops.aten.sub.Tensor(add_240, getitem_207);  getitem_207 = None
        mul_327 = torch.ops.aten.mul.Tensor(sub_101, rsqrt_40);  sub_101 = rsqrt_40 = None
        mul_328 = torch.ops.aten.mul.Tensor(mul_327, arg255_1);  mul_327 = arg255_1 = None
        add_242 = torch.ops.aten.add.Tensor(mul_328, arg256_1);  mul_328 = arg256_1 = None
        view_210 = torch.ops.aten.view.default(add_242, [512, 240]);  add_242 = None
        permute_129 = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_71 = torch.ops.aten.addmm.default(arg258_1, view_210, permute_129);  arg258_1 = view_210 = permute_129 = None
        view_211 = torch.ops.aten.view.default(addmm_71, [32, 16, 480]);  addmm_71 = None
        sigmoid_64 = torch.ops.aten.sigmoid.default(view_211)
        mul_329 = torch.ops.aten.mul.Tensor(view_211, sigmoid_64);  view_211 = sigmoid_64 = None
        view_212 = torch.ops.aten.view.default(mul_329, [512, 480]);  mul_329 = None
        permute_130 = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_72 = torch.ops.aten.addmm.default(arg260_1, view_212, permute_130);  arg260_1 = view_212 = permute_130 = None
        view_213 = torch.ops.aten.view.default(addmm_72, [32, 16, 240]);  addmm_72 = None
        add_243 = torch.ops.aten.add.Tensor(add_240, view_213);  add_240 = view_213 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_243, [2], correction = 0, keepdim = True)
        getitem_208 = var_mean_41[0]
        getitem_209 = var_mean_41[1];  var_mean_41 = None
        add_244 = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_41 = torch.ops.aten.rsqrt.default(add_244);  add_244 = None
        sub_102 = torch.ops.aten.sub.Tensor(add_243, getitem_209);  add_243 = getitem_209 = None
        mul_330 = torch.ops.aten.mul.Tensor(sub_102, rsqrt_41);  sub_102 = rsqrt_41 = None
        mul_331 = torch.ops.aten.mul.Tensor(mul_330, arg261_1);  mul_330 = arg261_1 = None
        add_245 = torch.ops.aten.add.Tensor(mul_331, arg262_1);  mul_331 = arg262_1 = None
        view_214 = torch.ops.aten.view.default(add_245, [8, 4, 16, -1]);  add_245 = None
        permute_131 = torch.ops.aten.permute.default(view_214, [0, 3, 2, 1]);  view_214 = None
        clone_77 = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
        view_215 = torch.ops.aten.view.default(clone_77, [7680, 4, 2, 2]);  clone_77 = None
        permute_132 = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
        clone_78 = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_216 = torch.ops.aten.view.default(clone_78, [8, 240, 8, 8]);  clone_78 = None
        convolution_67 = torch.ops.aten.convolution.default(view_216, arg263_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  view_216 = arg263_1 = None
        add_246 = torch.ops.aten.add.Tensor(arg265_1, 1e-05);  arg265_1 = None
        sqrt_61 = torch.ops.aten.sqrt.default(add_246);  add_246 = None
        reciprocal_61 = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_332 = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(mul_332, -1);  mul_332 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_103 = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_489);  convolution_67 = unsqueeze_489 = None
        mul_333 = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_491);  sub_103 = unsqueeze_491 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(arg266_1, -1);  arg266_1 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_334 = torch.ops.aten.mul.Tensor(mul_333, unsqueeze_493);  mul_333 = unsqueeze_493 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(arg267_1, -1);  arg267_1 = None
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_247 = torch.ops.aten.add.Tensor(mul_334, unsqueeze_495);  mul_334 = unsqueeze_495 = None
        sigmoid_65 = torch.ops.aten.sigmoid.default(add_247)
        mul_335 = torch.ops.aten.mul.Tensor(add_247, sigmoid_65);  add_247 = sigmoid_65 = None
        cat_5 = torch.ops.aten.cat.default([add_223, mul_335], 1);  add_223 = mul_335 = None
        convolution_68 = torch.ops.aten.convolution.default(cat_5, arg268_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  cat_5 = arg268_1 = None
        add_248 = torch.ops.aten.add.Tensor(arg270_1, 1e-05);  arg270_1 = None
        sqrt_62 = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_62 = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_336 = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_496 = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_497 = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        unsqueeze_498 = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_499 = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        sub_104 = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_497);  convolution_68 = unsqueeze_497 = None
        mul_337 = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_499);  sub_104 = unsqueeze_499 = None
        unsqueeze_500 = torch.ops.aten.unsqueeze.default(arg271_1, -1);  arg271_1 = None
        unsqueeze_501 = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_501);  mul_337 = unsqueeze_501 = None
        unsqueeze_502 = torch.ops.aten.unsqueeze.default(arg272_1, -1);  arg272_1 = None
        unsqueeze_503 = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_249 = torch.ops.aten.add.Tensor(mul_338, unsqueeze_503);  mul_338 = unsqueeze_503 = None
        sigmoid_66 = torch.ops.aten.sigmoid.default(add_249)
        mul_339 = torch.ops.aten.mul.Tensor(add_249, sigmoid_66);  add_249 = sigmoid_66 = None
        convolution_69 = torch.ops.aten.convolution.default(mul_339, arg273_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_339 = arg273_1 = None
        add_250 = torch.ops.aten.add.Tensor(arg275_1, 1e-05);  arg275_1 = None
        sqrt_63 = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_63 = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_340 = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_504 = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_505 = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        unsqueeze_506 = torch.ops.aten.unsqueeze.default(mul_340, -1);  mul_340 = None
        unsqueeze_507 = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        sub_105 = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_505);  convolution_69 = unsqueeze_505 = None
        mul_341 = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_507);  sub_105 = unsqueeze_507 = None
        unsqueeze_508 = torch.ops.aten.unsqueeze.default(arg276_1, -1);  arg276_1 = None
        unsqueeze_509 = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_342 = torch.ops.aten.mul.Tensor(mul_341, unsqueeze_509);  mul_341 = unsqueeze_509 = None
        unsqueeze_510 = torch.ops.aten.unsqueeze.default(arg277_1, -1);  arg277_1 = None
        unsqueeze_511 = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_251 = torch.ops.aten.add.Tensor(mul_342, unsqueeze_511);  mul_342 = unsqueeze_511 = None
        sigmoid_67 = torch.ops.aten.sigmoid.default(add_251)
        mul_343 = torch.ops.aten.mul.Tensor(add_251, sigmoid_67);  add_251 = sigmoid_67 = None
        mean_1 = torch.ops.aten.mean.dim(mul_343, [-1, -2], True);  mul_343 = None
        view_217 = torch.ops.aten.view.default(mean_1, [8, 640]);  mean_1 = None
        permute_133 = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
        addmm_73 = torch.ops.aten.addmm.default(arg279_1, view_217, permute_133);  arg279_1 = view_217 = permute_133 = None
        return (addmm_73,)
        
def load_args(reader):
    buf0 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf0, (16, 3, 3, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 6291456, device=device(type='cuda', index=0))
    reader.tensor(buf1, (8, 3, 256, 256), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf2, (16,), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf3, (16,), is_leaf=True)  # arg3_1
    buf4 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf4, (16,), is_leaf=True)  # arg4_1
    buf5 = reader.storage(None, 64, device=device(type='cuda', index=0))
    reader.tensor(buf5, (16,), is_leaf=True)  # arg5_1
    buf6 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf6, (64, 16, 1, 1), is_leaf=True)  # arg6_1
    buf7 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf7, (64,), is_leaf=True)  # arg7_1
    buf8 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf8, (64,), is_leaf=True)  # arg8_1
    buf9 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf9, (64,), is_leaf=True)  # arg9_1
    buf10 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf10, (64,), is_leaf=True)  # arg10_1
    buf11 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf11, (64, 1, 3, 3), is_leaf=True)  # arg11_1
    buf12 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf12, (64,), is_leaf=True)  # arg12_1
    buf13 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf13, (64,), is_leaf=True)  # arg13_1
    buf14 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf14, (64,), is_leaf=True)  # arg14_1
    buf15 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf15, (64,), is_leaf=True)  # arg15_1
    buf16 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf16, (32, 64, 1, 1), is_leaf=True)  # arg16_1
    buf17 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf17, (32,), is_leaf=True)  # arg17_1
    buf18 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf18, (32,), is_leaf=True)  # arg18_1
    buf19 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf19, (32,), is_leaf=True)  # arg19_1
    buf20 = reader.storage(None, 128, device=device(type='cuda', index=0))
    reader.tensor(buf20, (32,), is_leaf=True)  # arg20_1
    buf21 = reader.storage(None, 16384, device=device(type='cuda', index=0))
    reader.tensor(buf21, (128, 32, 1, 1), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf22, (128,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf23, (128,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf24, (128,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf25, (128,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 4608, device=device(type='cuda', index=0))
    reader.tensor(buf26, (128, 1, 3, 3), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf27, (128,), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf28, (128,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf29, (128,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf30, (128,), is_leaf=True)  # arg30_1
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
    buf36 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256, 64, 1, 1), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf37, (256,), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf38, (256,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256, 1, 3, 3), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256,), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf45, (256,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64, 256, 1, 1), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf47, (64,), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf48, (64,), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf49, (64,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf50, (64,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf51, (256, 64, 1, 1), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf52, (256,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf53, (256,), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf54, (256,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf56, (256, 1, 3, 3), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256,), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf59, (256,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf60, (256,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf61, (64, 256, 1, 1), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf62, (64,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf63, (64,), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf64, (64,), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf65, (64,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf66, (256, 64, 1, 1), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf67, (256,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf68, (256,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf69, (256,), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf70, (256,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 9216, device=device(type='cuda', index=0))
    reader.tensor(buf71, (256, 1, 3, 3), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf72, (256,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf73, (256,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf74, (256,), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf75, (256,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf76, (96, 256, 1, 1), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf77, (96,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf78, (96,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf79, (96,), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf80, (96,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 331776, device=device(type='cuda', index=0))
    reader.tensor(buf81, (96, 96, 3, 3), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf82, (96,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf83, (96,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf84, (96,), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf85, (96,), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 55296, device=device(type='cuda', index=0))
    reader.tensor(buf86, (144, 96, 1, 1), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf87, (144,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf88, (144,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 248832, device=device(type='cuda', index=0))
    reader.tensor(buf89, (432, 144), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf90, (432,), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf91, (144, 144), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf92, (144,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf93, (144,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf94, (144,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf95, (288, 144), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf96, (288,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf97, (144, 288), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf98, (144,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf99, (144,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf100, (144,), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 248832, device=device(type='cuda', index=0))
    reader.tensor(buf101, (432, 144), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 1728, device=device(type='cuda', index=0))
    reader.tensor(buf102, (432,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 82944, device=device(type='cuda', index=0))
    reader.tensor(buf103, (144, 144), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf104, (144,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf105, (144,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf106, (144,), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf107, (288, 144), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 1152, device=device(type='cuda', index=0))
    reader.tensor(buf108, (288,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 165888, device=device(type='cuda', index=0))
    reader.tensor(buf109, (144, 288), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf110, (144,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf111, (144,), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 576, device=device(type='cuda', index=0))
    reader.tensor(buf112, (144,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 55296, device=device(type='cuda', index=0))
    reader.tensor(buf113, (96, 144, 1, 1), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf114, (96,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf115, (96,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf116, (96,), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf117, (96,), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 663552, device=device(type='cuda', index=0))
    reader.tensor(buf118, (96, 192, 3, 3), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf119, (96,), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf120, (96,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf121, (96,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 384, device=device(type='cuda', index=0))
    reader.tensor(buf122, (96,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf123, (384, 96, 1, 1), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf124, (384,), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf125, (384,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf126, (384,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf127, (384,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 13824, device=device(type='cuda', index=0))
    reader.tensor(buf128, (384, 1, 3, 3), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf129, (384,), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf130, (384,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf131, (384,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf132, (384,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 196608, device=device(type='cuda', index=0))
    reader.tensor(buf133, (128, 384, 1, 1), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf134, (128,), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf135, (128,), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf136, (128,), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf137, (128,), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 589824, device=device(type='cuda', index=0))
    reader.tensor(buf138, (128, 128, 3, 3), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf139, (128,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf140, (128,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf141, (128,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf142, (128,), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf143, (192, 128, 1, 1), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf144, (192,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf145, (192,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf146, (576, 192), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf147, (576,), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf148, (192, 192), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf149, (192,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf150, (192,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf151, (192,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf152, (384, 192), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf153, (384,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf154, (192, 384), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf155, (192,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf156, (192,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf157, (192,), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf158, (576, 192), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf159, (576,), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf160, (192, 192), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf161, (192,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf162, (192,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf163, (192,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf164, (384, 192), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf165, (384,), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf166, (192, 384), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf167, (192,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf168, (192,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf169, (192,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf170, (576, 192), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf171, (576,), is_leaf=True)  # arg171_1
    buf172 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf172, (192, 192), is_leaf=True)  # arg172_1
    buf173 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf173, (192,), is_leaf=True)  # arg173_1
    buf174 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf174, (192,), is_leaf=True)  # arg174_1
    buf175 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf175, (192,), is_leaf=True)  # arg175_1
    buf176 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf176, (384, 192), is_leaf=True)  # arg176_1
    buf177 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf177, (384,), is_leaf=True)  # arg177_1
    buf178 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf178, (192, 384), is_leaf=True)  # arg178_1
    buf179 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf179, (192,), is_leaf=True)  # arg179_1
    buf180 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf180, (192,), is_leaf=True)  # arg180_1
    buf181 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf181, (192,), is_leaf=True)  # arg181_1
    buf182 = reader.storage(None, 442368, device=device(type='cuda', index=0))
    reader.tensor(buf182, (576, 192), is_leaf=True)  # arg182_1
    buf183 = reader.storage(None, 2304, device=device(type='cuda', index=0))
    reader.tensor(buf183, (576,), is_leaf=True)  # arg183_1
    buf184 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf184, (192, 192), is_leaf=True)  # arg184_1
    buf185 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf185, (192,), is_leaf=True)  # arg185_1
    buf186 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf186, (192,), is_leaf=True)  # arg186_1
    buf187 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf187, (192,), is_leaf=True)  # arg187_1
    buf188 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf188, (384, 192), is_leaf=True)  # arg188_1
    buf189 = reader.storage(None, 1536, device=device(type='cuda', index=0))
    reader.tensor(buf189, (384,), is_leaf=True)  # arg189_1
    buf190 = reader.storage(None, 294912, device=device(type='cuda', index=0))
    reader.tensor(buf190, (192, 384), is_leaf=True)  # arg190_1
    buf191 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf191, (192,), is_leaf=True)  # arg191_1
    buf192 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf192, (192,), is_leaf=True)  # arg192_1
    buf193 = reader.storage(None, 768, device=device(type='cuda', index=0))
    reader.tensor(buf193, (192,), is_leaf=True)  # arg193_1
    buf194 = reader.storage(None, 98304, device=device(type='cuda', index=0))
    reader.tensor(buf194, (128, 192, 1, 1), is_leaf=True)  # arg194_1
    buf195 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf195, (128,), is_leaf=True)  # arg195_1
    buf196 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf196, (128,), is_leaf=True)  # arg196_1
    buf197 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf197, (128,), is_leaf=True)  # arg197_1
    buf198 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf198, (128,), is_leaf=True)  # arg198_1
    buf199 = reader.storage(None, 1179648, device=device(type='cuda', index=0))
    reader.tensor(buf199, (128, 256, 3, 3), is_leaf=True)  # arg199_1
    buf200 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf200, (128,), is_leaf=True)  # arg200_1
    buf201 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf201, (128,), is_leaf=True)  # arg201_1
    buf202 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf202, (128,), is_leaf=True)  # arg202_1
    buf203 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf203, (128,), is_leaf=True)  # arg203_1
    buf204 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf204, (512, 128, 1, 1), is_leaf=True)  # arg204_1
    buf205 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf205, (512,), is_leaf=True)  # arg205_1
    buf206 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf206, (512,), is_leaf=True)  # arg206_1
    buf207 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf207, (512,), is_leaf=True)  # arg207_1
    buf208 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf208, (512,), is_leaf=True)  # arg208_1
    buf209 = reader.storage(None, 18432, device=device(type='cuda', index=0))
    reader.tensor(buf209, (512, 1, 3, 3), is_leaf=True)  # arg209_1
    buf210 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf210, (512,), is_leaf=True)  # arg210_1
    buf211 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf211, (512,), is_leaf=True)  # arg211_1
    buf212 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf212, (512,), is_leaf=True)  # arg212_1
    buf213 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf213, (512,), is_leaf=True)  # arg213_1
    buf214 = reader.storage(None, 327680, device=device(type='cuda', index=0))
    reader.tensor(buf214, (160, 512, 1, 1), is_leaf=True)  # arg214_1
    buf215 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf215, (160,), is_leaf=True)  # arg215_1
    buf216 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf216, (160,), is_leaf=True)  # arg216_1
    buf217 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf217, (160,), is_leaf=True)  # arg217_1
    buf218 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf218, (160,), is_leaf=True)  # arg218_1
    buf219 = reader.storage(None, 921600, device=device(type='cuda', index=0))
    reader.tensor(buf219, (160, 160, 3, 3), is_leaf=True)  # arg219_1
    buf220 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf220, (160,), is_leaf=True)  # arg220_1
    buf221 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf221, (160,), is_leaf=True)  # arg221_1
    buf222 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf222, (160,), is_leaf=True)  # arg222_1
    buf223 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf223, (160,), is_leaf=True)  # arg223_1
    buf224 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf224, (240, 160, 1, 1), is_leaf=True)  # arg224_1
    buf225 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf225, (240,), is_leaf=True)  # arg225_1
    buf226 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf226, (240,), is_leaf=True)  # arg226_1
    buf227 = reader.storage(None, 691200, device=device(type='cuda', index=0))
    reader.tensor(buf227, (720, 240), is_leaf=True)  # arg227_1
    buf228 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf228, (720,), is_leaf=True)  # arg228_1
    buf229 = reader.storage(None, 230400, device=device(type='cuda', index=0))
    reader.tensor(buf229, (240, 240), is_leaf=True)  # arg229_1
    buf230 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf230, (240,), is_leaf=True)  # arg230_1
    buf231 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf231, (240,), is_leaf=True)  # arg231_1
    buf232 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf232, (240,), is_leaf=True)  # arg232_1
    buf233 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf233, (480, 240), is_leaf=True)  # arg233_1
    buf234 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf234, (480,), is_leaf=True)  # arg234_1
    buf235 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf235, (240, 480), is_leaf=True)  # arg235_1
    buf236 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf236, (240,), is_leaf=True)  # arg236_1
    buf237 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf237, (240,), is_leaf=True)  # arg237_1
    buf238 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf238, (240,), is_leaf=True)  # arg238_1
    buf239 = reader.storage(None, 691200, device=device(type='cuda', index=0))
    reader.tensor(buf239, (720, 240), is_leaf=True)  # arg239_1
    buf240 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf240, (720,), is_leaf=True)  # arg240_1
    buf241 = reader.storage(None, 230400, device=device(type='cuda', index=0))
    reader.tensor(buf241, (240, 240), is_leaf=True)  # arg241_1
    buf242 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf242, (240,), is_leaf=True)  # arg242_1
    buf243 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf243, (240,), is_leaf=True)  # arg243_1
    buf244 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf244, (240,), is_leaf=True)  # arg244_1
    buf245 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf245, (480, 240), is_leaf=True)  # arg245_1
    buf246 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf246, (480,), is_leaf=True)  # arg246_1
    buf247 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf247, (240, 480), is_leaf=True)  # arg247_1
    buf248 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf248, (240,), is_leaf=True)  # arg248_1
    buf249 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf249, (240,), is_leaf=True)  # arg249_1
    buf250 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf250, (240,), is_leaf=True)  # arg250_1
    buf251 = reader.storage(None, 691200, device=device(type='cuda', index=0))
    reader.tensor(buf251, (720, 240), is_leaf=True)  # arg251_1
    buf252 = reader.storage(None, 2880, device=device(type='cuda', index=0))
    reader.tensor(buf252, (720,), is_leaf=True)  # arg252_1
    buf253 = reader.storage(None, 230400, device=device(type='cuda', index=0))
    reader.tensor(buf253, (240, 240), is_leaf=True)  # arg253_1
    buf254 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf254, (240,), is_leaf=True)  # arg254_1
    buf255 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf255, (240,), is_leaf=True)  # arg255_1
    buf256 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf256, (240,), is_leaf=True)  # arg256_1
    buf257 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf257, (480, 240), is_leaf=True)  # arg257_1
    buf258 = reader.storage(None, 1920, device=device(type='cuda', index=0))
    reader.tensor(buf258, (480,), is_leaf=True)  # arg258_1
    buf259 = reader.storage(None, 460800, device=device(type='cuda', index=0))
    reader.tensor(buf259, (240, 480), is_leaf=True)  # arg259_1
    buf260 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf260, (240,), is_leaf=True)  # arg260_1
    buf261 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf261, (240,), is_leaf=True)  # arg261_1
    buf262 = reader.storage(None, 960, device=device(type='cuda', index=0))
    reader.tensor(buf262, (240,), is_leaf=True)  # arg262_1
    buf263 = reader.storage(None, 153600, device=device(type='cuda', index=0))
    reader.tensor(buf263, (160, 240, 1, 1), is_leaf=True)  # arg263_1
    buf264 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf264, (160,), is_leaf=True)  # arg264_1
    buf265 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf265, (160,), is_leaf=True)  # arg265_1
    buf266 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf266, (160,), is_leaf=True)  # arg266_1
    buf267 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf267, (160,), is_leaf=True)  # arg267_1
    buf268 = reader.storage(None, 1843200, device=device(type='cuda', index=0))
    reader.tensor(buf268, (160, 320, 3, 3), is_leaf=True)  # arg268_1
    buf269 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf269, (160,), is_leaf=True)  # arg269_1
    buf270 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf270, (160,), is_leaf=True)  # arg270_1
    buf271 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf271, (160,), is_leaf=True)  # arg271_1
    buf272 = reader.storage(None, 640, device=device(type='cuda', index=0))
    reader.tensor(buf272, (160,), is_leaf=True)  # arg272_1
    buf273 = reader.storage(None, 409600, device=device(type='cuda', index=0))
    reader.tensor(buf273, (640, 160, 1, 1), is_leaf=True)  # arg273_1
    buf274 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf274, (640,), is_leaf=True)  # arg274_1
    buf275 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf275, (640,), is_leaf=True)  # arg275_1
    buf276 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf276, (640,), is_leaf=True)  # arg276_1
    buf277 = reader.storage(None, 2560, device=device(type='cuda', index=0))
    reader.tensor(buf277, (640,), is_leaf=True)  # arg277_1
    buf278 = reader.storage(None, 2560000, device=device(type='cuda', index=0))
    reader.tensor(buf278, (1000, 640), is_leaf=True)  # arg278_1
    buf279 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf279, (1000,), is_leaf=True)  # arg279_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)