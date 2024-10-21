
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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1):
        convolution_39 = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None
        add_76 = torch.ops.aten.add.Tensor(arg3_1, 1e-05);  arg3_1 = None
        sqrt_31 = torch.ops.aten.sqrt.default(add_76);  add_76 = None
        reciprocal_31 = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_128 = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(mul_128, -1);  mul_128 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        sub_34 = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_249);  convolution_39 = unsqueeze_249 = None
        mul_129 = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_251);  sub_34 = unsqueeze_251 = None
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_130 = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_253);  mul_129 = unsqueeze_253 = None
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_77 = torch.ops.aten.add.Tensor(mul_130, unsqueeze_255);  mul_130 = unsqueeze_255 = None
        sigmoid_32 = torch.ops.aten.sigmoid.default(add_77)
        mul_131 = torch.ops.aten.mul.Tensor(add_77, sigmoid_32);  add_77 = sigmoid_32 = None
        convolution_40 = torch.ops.aten.convolution.default(mul_131, arg6_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_131 = arg6_1 = None
        add_78 = torch.ops.aten.add.Tensor(arg8_1, 1e-05);  arg8_1 = None
        sqrt_32 = torch.ops.aten.sqrt.default(add_78);  add_78 = None
        reciprocal_32 = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_132 = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_256 = torch.ops.aten.unsqueeze.default(arg7_1, -1);  arg7_1 = None
        unsqueeze_257 = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        unsqueeze_258 = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
        unsqueeze_259 = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        sub_35 = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_257);  convolution_40 = unsqueeze_257 = None
        mul_133 = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_259);  sub_35 = unsqueeze_259 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_261);  mul_133 = unsqueeze_261 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_263 = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_79 = torch.ops.aten.add.Tensor(mul_134, unsqueeze_263);  mul_134 = unsqueeze_263 = None
        sigmoid_33 = torch.ops.aten.sigmoid.default(add_79)
        mul_135 = torch.ops.aten.mul.Tensor(add_79, sigmoid_33);  add_79 = sigmoid_33 = None
        convolution_41 = torch.ops.aten.convolution.default(mul_135, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  mul_135 = arg11_1 = None
        add_80 = torch.ops.aten.add.Tensor(arg13_1, 1e-05);  arg13_1 = None
        sqrt_33 = torch.ops.aten.sqrt.default(add_80);  add_80 = None
        reciprocal_33 = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_136 = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_264 = torch.ops.aten.unsqueeze.default(arg12_1, -1);  arg12_1 = None
        unsqueeze_265 = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        unsqueeze_266 = torch.ops.aten.unsqueeze.default(mul_136, -1);  mul_136 = None
        unsqueeze_267 = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        sub_36 = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_265);  convolution_41 = unsqueeze_265 = None
        mul_137 = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_267);  sub_36 = unsqueeze_267 = None
        unsqueeze_268 = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_269 = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_138 = torch.ops.aten.mul.Tensor(mul_137, unsqueeze_269);  mul_137 = unsqueeze_269 = None
        unsqueeze_270 = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_271 = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_81 = torch.ops.aten.add.Tensor(mul_138, unsqueeze_271);  mul_138 = unsqueeze_271 = None
        sigmoid_34 = torch.ops.aten.sigmoid.default(add_81)
        mul_139 = torch.ops.aten.mul.Tensor(add_81, sigmoid_34);  add_81 = sigmoid_34 = None
        _low_memory_max_pool2d_with_offsets_1 = torch.ops.prims._low_memory_max_pool2d_with_offsets.default(mul_139, [3, 3], [2, 2], [1, 1], [1, 1], False);  mul_139 = None
        getitem_8 = _low_memory_max_pool2d_with_offsets_1[0];  _low_memory_max_pool2d_with_offsets_1 = None
        convolution_42 = torch.ops.aten.convolution.default(getitem_8, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg16_1 = None
        add_82 = torch.ops.aten.add.Tensor(arg18_1, 1e-05);  arg18_1 = None
        sqrt_34 = torch.ops.aten.sqrt.default(add_82);  add_82 = None
        reciprocal_34 = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_140 = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_272 = torch.ops.aten.unsqueeze.default(arg17_1, -1);  arg17_1 = None
        unsqueeze_273 = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        unsqueeze_274 = torch.ops.aten.unsqueeze.default(mul_140, -1);  mul_140 = None
        unsqueeze_275 = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        sub_37 = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_273);  convolution_42 = unsqueeze_273 = None
        mul_141 = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_275);  sub_37 = unsqueeze_275 = None
        unsqueeze_276 = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_277 = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_142 = torch.ops.aten.mul.Tensor(mul_141, unsqueeze_277);  mul_141 = unsqueeze_277 = None
        unsqueeze_278 = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_279 = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_83 = torch.ops.aten.add.Tensor(mul_142, unsqueeze_279);  mul_142 = unsqueeze_279 = None
        sigmoid_35 = torch.ops.aten.sigmoid.default(add_83)
        mul_143 = torch.ops.aten.mul.Tensor(add_83, sigmoid_35);  add_83 = sigmoid_35 = None
        convolution_43 = torch.ops.aten.convolution.default(mul_143, arg21_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4);  mul_143 = arg21_1 = None
        add_84 = torch.ops.aten.add.Tensor(arg23_1, 1e-05);  arg23_1 = None
        sqrt_35 = torch.ops.aten.sqrt.default(add_84);  add_84 = None
        reciprocal_35 = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_144 = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_280 = torch.ops.aten.unsqueeze.default(arg22_1, -1);  arg22_1 = None
        unsqueeze_281 = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        unsqueeze_282 = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
        unsqueeze_283 = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        sub_38 = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_281);  convolution_43 = unsqueeze_281 = None
        mul_145 = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_283);  sub_38 = unsqueeze_283 = None
        unsqueeze_284 = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_285 = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_285);  mul_145 = unsqueeze_285 = None
        unsqueeze_286 = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_287 = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_85 = torch.ops.aten.add.Tensor(mul_146, unsqueeze_287);  mul_146 = unsqueeze_287 = None
        sigmoid_36 = torch.ops.aten.sigmoid.default(add_85)
        mul_147 = torch.ops.aten.mul.Tensor(add_85, sigmoid_36);  add_85 = sigmoid_36 = None
        mean_6 = torch.ops.aten.mean.dim(mul_147, [2, 3])
        view_86 = torch.ops.aten.view.default(mean_6, [8, 1, -1]);  mean_6 = None
        convolution_44 = torch.ops.aten.convolution.default(view_86, arg26_1, None, [1], [1], [1], False, [0], 1);  view_86 = arg26_1 = None
        sigmoid_37 = torch.ops.aten.sigmoid.default(convolution_44);  convolution_44 = None
        view_87 = torch.ops.aten.view.default(sigmoid_37, [8, -1, 1, 1]);  sigmoid_37 = None
        expand_23 = torch.ops.aten.expand.default(view_87, [8, 64, 64, 64]);  view_87 = None
        mul_148 = torch.ops.aten.mul.Tensor(mul_147, expand_23);  mul_147 = expand_23 = None
        convolution_45 = torch.ops.aten.convolution.default(mul_148, arg27_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_148 = arg27_1 = None
        add_86 = torch.ops.aten.add.Tensor(arg29_1, 1e-05);  arg29_1 = None
        sqrt_36 = torch.ops.aten.sqrt.default(add_86);  add_86 = None
        reciprocal_36 = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_149 = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_288 = torch.ops.aten.unsqueeze.default(arg28_1, -1);  arg28_1 = None
        unsqueeze_289 = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        unsqueeze_290 = torch.ops.aten.unsqueeze.default(mul_149, -1);  mul_149 = None
        unsqueeze_291 = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        sub_39 = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_289);  convolution_45 = unsqueeze_289 = None
        mul_150 = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_291);  sub_39 = unsqueeze_291 = None
        unsqueeze_292 = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_293 = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_151 = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_293);  mul_150 = unsqueeze_293 = None
        unsqueeze_294 = torch.ops.aten.unsqueeze.default(arg31_1, -1);  arg31_1 = None
        unsqueeze_295 = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_87 = torch.ops.aten.add.Tensor(mul_151, unsqueeze_295);  mul_151 = unsqueeze_295 = None
        convolution_46 = torch.ops.aten.convolution.default(getitem_8, arg32_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem_8 = arg32_1 = None
        add_88 = torch.ops.aten.add.Tensor(arg34_1, 1e-05);  arg34_1 = None
        sqrt_37 = torch.ops.aten.sqrt.default(add_88);  add_88 = None
        reciprocal_37 = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_152 = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_296 = torch.ops.aten.unsqueeze.default(arg33_1, -1);  arg33_1 = None
        unsqueeze_297 = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        unsqueeze_298 = torch.ops.aten.unsqueeze.default(mul_152, -1);  mul_152 = None
        unsqueeze_299 = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        sub_40 = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_297);  convolution_46 = unsqueeze_297 = None
        mul_153 = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_299);  sub_40 = unsqueeze_299 = None
        unsqueeze_300 = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_301 = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_154 = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_301);  mul_153 = unsqueeze_301 = None
        unsqueeze_302 = torch.ops.aten.unsqueeze.default(arg36_1, -1);  arg36_1 = None
        unsqueeze_303 = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_89 = torch.ops.aten.add.Tensor(mul_154, unsqueeze_303);  mul_154 = unsqueeze_303 = None
        add_90 = torch.ops.aten.add.Tensor(add_87, add_89);  add_87 = add_89 = None
        sigmoid_38 = torch.ops.aten.sigmoid.default(add_90)
        mul_155 = torch.ops.aten.mul.Tensor(add_90, sigmoid_38);  add_90 = sigmoid_38 = None
        convolution_47 = torch.ops.aten.convolution.default(mul_155, arg37_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg37_1 = None
        add_91 = torch.ops.aten.add.Tensor(arg39_1, 1e-05);  arg39_1 = None
        sqrt_38 = torch.ops.aten.sqrt.default(add_91);  add_91 = None
        reciprocal_38 = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_156 = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_304 = torch.ops.aten.unsqueeze.default(arg38_1, -1);  arg38_1 = None
        unsqueeze_305 = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        unsqueeze_306 = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
        unsqueeze_307 = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        sub_41 = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_305);  convolution_47 = unsqueeze_305 = None
        mul_157 = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_307);  sub_41 = unsqueeze_307 = None
        unsqueeze_308 = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_309 = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_309);  mul_157 = unsqueeze_309 = None
        unsqueeze_310 = torch.ops.aten.unsqueeze.default(arg41_1, -1);  arg41_1 = None
        unsqueeze_311 = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_92 = torch.ops.aten.add.Tensor(mul_158, unsqueeze_311);  mul_158 = unsqueeze_311 = None
        sigmoid_39 = torch.ops.aten.sigmoid.default(add_92)
        mul_159 = torch.ops.aten.mul.Tensor(add_92, sigmoid_39);  add_92 = sigmoid_39 = None
        convolution_48 = torch.ops.aten.convolution.default(mul_159, arg42_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4);  mul_159 = arg42_1 = None
        add_93 = torch.ops.aten.add.Tensor(arg44_1, 1e-05);  arg44_1 = None
        sqrt_39 = torch.ops.aten.sqrt.default(add_93);  add_93 = None
        reciprocal_39 = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_160 = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_312 = torch.ops.aten.unsqueeze.default(arg43_1, -1);  arg43_1 = None
        unsqueeze_313 = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        unsqueeze_314 = torch.ops.aten.unsqueeze.default(mul_160, -1);  mul_160 = None
        unsqueeze_315 = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        sub_42 = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_313);  convolution_48 = unsqueeze_313 = None
        mul_161 = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_315);  sub_42 = unsqueeze_315 = None
        unsqueeze_316 = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_317 = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_162 = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_317);  mul_161 = unsqueeze_317 = None
        unsqueeze_318 = torch.ops.aten.unsqueeze.default(arg46_1, -1);  arg46_1 = None
        unsqueeze_319 = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_94 = torch.ops.aten.add.Tensor(mul_162, unsqueeze_319);  mul_162 = unsqueeze_319 = None
        sigmoid_40 = torch.ops.aten.sigmoid.default(add_94)
        mul_163 = torch.ops.aten.mul.Tensor(add_94, sigmoid_40);  add_94 = sigmoid_40 = None
        mean_7 = torch.ops.aten.mean.dim(mul_163, [2, 3])
        view_88 = torch.ops.aten.view.default(mean_7, [8, 1, -1]);  mean_7 = None
        convolution_49 = torch.ops.aten.convolution.default(view_88, arg47_1, None, [1], [1], [1], False, [0], 1);  view_88 = arg47_1 = None
        sigmoid_41 = torch.ops.aten.sigmoid.default(convolution_49);  convolution_49 = None
        view_89 = torch.ops.aten.view.default(sigmoid_41, [8, -1, 1, 1]);  sigmoid_41 = None
        expand_24 = torch.ops.aten.expand.default(view_89, [8, 64, 64, 64]);  view_89 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, expand_24);  mul_163 = expand_24 = None
        convolution_50 = torch.ops.aten.convolution.default(mul_164, arg48_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_164 = arg48_1 = None
        add_95 = torch.ops.aten.add.Tensor(arg50_1, 1e-05);  arg50_1 = None
        sqrt_40 = torch.ops.aten.sqrt.default(add_95);  add_95 = None
        reciprocal_40 = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_165 = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_320 = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_321 = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        unsqueeze_322 = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
        unsqueeze_323 = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        sub_43 = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_321);  convolution_50 = unsqueeze_321 = None
        mul_166 = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_323);  sub_43 = unsqueeze_323 = None
        unsqueeze_324 = torch.ops.aten.unsqueeze.default(arg51_1, -1);  arg51_1 = None
        unsqueeze_325 = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_325);  mul_166 = unsqueeze_325 = None
        unsqueeze_326 = torch.ops.aten.unsqueeze.default(arg52_1, -1);  arg52_1 = None
        unsqueeze_327 = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_96 = torch.ops.aten.add.Tensor(mul_167, unsqueeze_327);  mul_167 = unsqueeze_327 = None
        add_97 = torch.ops.aten.add.Tensor(add_96, mul_155);  add_96 = mul_155 = None
        sigmoid_42 = torch.ops.aten.sigmoid.default(add_97)
        mul_168 = torch.ops.aten.mul.Tensor(add_97, sigmoid_42);  add_97 = sigmoid_42 = None
        convolution_51 = torch.ops.aten.convolution.default(mul_168, arg53_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg53_1 = None
        add_98 = torch.ops.aten.add.Tensor(arg55_1, 1e-05);  arg55_1 = None
        sqrt_41 = torch.ops.aten.sqrt.default(add_98);  add_98 = None
        reciprocal_41 = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_169 = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_328 = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_329 = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        unsqueeze_330 = torch.ops.aten.unsqueeze.default(mul_169, -1);  mul_169 = None
        unsqueeze_331 = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        sub_44 = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_329);  convolution_51 = unsqueeze_329 = None
        mul_170 = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_331);  sub_44 = unsqueeze_331 = None
        unsqueeze_332 = torch.ops.aten.unsqueeze.default(arg56_1, -1);  arg56_1 = None
        unsqueeze_333 = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_171 = torch.ops.aten.mul.Tensor(mul_170, unsqueeze_333);  mul_170 = unsqueeze_333 = None
        unsqueeze_334 = torch.ops.aten.unsqueeze.default(arg57_1, -1);  arg57_1 = None
        unsqueeze_335 = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_99 = torch.ops.aten.add.Tensor(mul_171, unsqueeze_335);  mul_171 = unsqueeze_335 = None
        sigmoid_43 = torch.ops.aten.sigmoid.default(add_99)
        mul_172 = torch.ops.aten.mul.Tensor(add_99, sigmoid_43);  add_99 = sigmoid_43 = None
        convolution_52 = torch.ops.aten.convolution.default(mul_172, arg58_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8);  mul_172 = arg58_1 = None
        add_100 = torch.ops.aten.add.Tensor(arg60_1, 1e-05);  arg60_1 = None
        sqrt_42 = torch.ops.aten.sqrt.default(add_100);  add_100 = None
        reciprocal_42 = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_173 = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_336 = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_337 = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        unsqueeze_338 = torch.ops.aten.unsqueeze.default(mul_173, -1);  mul_173 = None
        unsqueeze_339 = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        sub_45 = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_337);  convolution_52 = unsqueeze_337 = None
        mul_174 = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_339);  sub_45 = unsqueeze_339 = None
        unsqueeze_340 = torch.ops.aten.unsqueeze.default(arg61_1, -1);  arg61_1 = None
        unsqueeze_341 = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_175 = torch.ops.aten.mul.Tensor(mul_174, unsqueeze_341);  mul_174 = unsqueeze_341 = None
        unsqueeze_342 = torch.ops.aten.unsqueeze.default(arg62_1, -1);  arg62_1 = None
        unsqueeze_343 = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_101 = torch.ops.aten.add.Tensor(mul_175, unsqueeze_343);  mul_175 = unsqueeze_343 = None
        sigmoid_44 = torch.ops.aten.sigmoid.default(add_101)
        mul_176 = torch.ops.aten.mul.Tensor(add_101, sigmoid_44);  add_101 = sigmoid_44 = None
        mean_8 = torch.ops.aten.mean.dim(mul_176, [2, 3])
        view_90 = torch.ops.aten.view.default(mean_8, [8, 1, -1]);  mean_8 = None
        convolution_53 = torch.ops.aten.convolution.default(view_90, arg63_1, None, [1], [2], [1], False, [0], 1);  view_90 = arg63_1 = None
        sigmoid_45 = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
        view_91 = torch.ops.aten.view.default(sigmoid_45, [8, -1, 1, 1]);  sigmoid_45 = None
        expand_25 = torch.ops.aten.expand.default(view_91, [8, 128, 32, 32]);  view_91 = None
        mul_177 = torch.ops.aten.mul.Tensor(mul_176, expand_25);  mul_176 = expand_25 = None
        convolution_54 = torch.ops.aten.convolution.default(mul_177, arg64_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_177 = arg64_1 = None
        add_102 = torch.ops.aten.add.Tensor(arg66_1, 1e-05);  arg66_1 = None
        sqrt_43 = torch.ops.aten.sqrt.default(add_102);  add_102 = None
        reciprocal_43 = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_178 = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_344 = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_345 = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        unsqueeze_346 = torch.ops.aten.unsqueeze.default(mul_178, -1);  mul_178 = None
        unsqueeze_347 = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        sub_46 = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_345);  convolution_54 = unsqueeze_345 = None
        mul_179 = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_347);  sub_46 = unsqueeze_347 = None
        unsqueeze_348 = torch.ops.aten.unsqueeze.default(arg67_1, -1);  arg67_1 = None
        unsqueeze_349 = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_180 = torch.ops.aten.mul.Tensor(mul_179, unsqueeze_349);  mul_179 = unsqueeze_349 = None
        unsqueeze_350 = torch.ops.aten.unsqueeze.default(arg68_1, -1);  arg68_1 = None
        unsqueeze_351 = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_103 = torch.ops.aten.add.Tensor(mul_180, unsqueeze_351);  mul_180 = unsqueeze_351 = None
        convolution_55 = torch.ops.aten.convolution.default(mul_168, arg69_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_168 = arg69_1 = None
        add_104 = torch.ops.aten.add.Tensor(arg71_1, 1e-05);  arg71_1 = None
        sqrt_44 = torch.ops.aten.sqrt.default(add_104);  add_104 = None
        reciprocal_44 = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_181 = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_352 = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_353 = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        unsqueeze_354 = torch.ops.aten.unsqueeze.default(mul_181, -1);  mul_181 = None
        unsqueeze_355 = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        sub_47 = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_353);  convolution_55 = unsqueeze_353 = None
        mul_182 = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_355);  sub_47 = unsqueeze_355 = None
        unsqueeze_356 = torch.ops.aten.unsqueeze.default(arg72_1, -1);  arg72_1 = None
        unsqueeze_357 = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_183 = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_357);  mul_182 = unsqueeze_357 = None
        unsqueeze_358 = torch.ops.aten.unsqueeze.default(arg73_1, -1);  arg73_1 = None
        unsqueeze_359 = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_105 = torch.ops.aten.add.Tensor(mul_183, unsqueeze_359);  mul_183 = unsqueeze_359 = None
        add_106 = torch.ops.aten.add.Tensor(add_103, add_105);  add_103 = add_105 = None
        sigmoid_46 = torch.ops.aten.sigmoid.default(add_106)
        mul_184 = torch.ops.aten.mul.Tensor(add_106, sigmoid_46);  add_106 = sigmoid_46 = None
        convolution_56 = torch.ops.aten.convolution.default(mul_184, arg74_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg74_1 = None
        add_107 = torch.ops.aten.add.Tensor(arg76_1, 1e-05);  arg76_1 = None
        sqrt_45 = torch.ops.aten.sqrt.default(add_107);  add_107 = None
        reciprocal_45 = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_185 = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_360 = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_361 = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        unsqueeze_362 = torch.ops.aten.unsqueeze.default(mul_185, -1);  mul_185 = None
        unsqueeze_363 = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        sub_48 = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_361);  convolution_56 = unsqueeze_361 = None
        mul_186 = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_363);  sub_48 = unsqueeze_363 = None
        unsqueeze_364 = torch.ops.aten.unsqueeze.default(arg77_1, -1);  arg77_1 = None
        unsqueeze_365 = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_187 = torch.ops.aten.mul.Tensor(mul_186, unsqueeze_365);  mul_186 = unsqueeze_365 = None
        unsqueeze_366 = torch.ops.aten.unsqueeze.default(arg78_1, -1);  arg78_1 = None
        unsqueeze_367 = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_108 = torch.ops.aten.add.Tensor(mul_187, unsqueeze_367);  mul_187 = unsqueeze_367 = None
        sigmoid_47 = torch.ops.aten.sigmoid.default(add_108)
        mul_188 = torch.ops.aten.mul.Tensor(add_108, sigmoid_47);  add_108 = sigmoid_47 = None
        convolution_57 = torch.ops.aten.convolution.default(mul_188, arg79_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8);  mul_188 = arg79_1 = None
        add_109 = torch.ops.aten.add.Tensor(arg81_1, 1e-05);  arg81_1 = None
        sqrt_46 = torch.ops.aten.sqrt.default(add_109);  add_109 = None
        reciprocal_46 = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_189 = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_368 = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_369 = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        unsqueeze_370 = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
        unsqueeze_371 = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        sub_49 = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_369);  convolution_57 = unsqueeze_369 = None
        mul_190 = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_371);  sub_49 = unsqueeze_371 = None
        unsqueeze_372 = torch.ops.aten.unsqueeze.default(arg82_1, -1);  arg82_1 = None
        unsqueeze_373 = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_373);  mul_190 = unsqueeze_373 = None
        unsqueeze_374 = torch.ops.aten.unsqueeze.default(arg83_1, -1);  arg83_1 = None
        unsqueeze_375 = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_110 = torch.ops.aten.add.Tensor(mul_191, unsqueeze_375);  mul_191 = unsqueeze_375 = None
        sigmoid_48 = torch.ops.aten.sigmoid.default(add_110)
        mul_192 = torch.ops.aten.mul.Tensor(add_110, sigmoid_48);  add_110 = sigmoid_48 = None
        mean_9 = torch.ops.aten.mean.dim(mul_192, [2, 3])
        view_92 = torch.ops.aten.view.default(mean_9, [8, 1, -1]);  mean_9 = None
        convolution_58 = torch.ops.aten.convolution.default(view_92, arg84_1, None, [1], [2], [1], False, [0], 1);  view_92 = arg84_1 = None
        sigmoid_49 = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
        view_93 = torch.ops.aten.view.default(sigmoid_49, [8, -1, 1, 1]);  sigmoid_49 = None
        expand_26 = torch.ops.aten.expand.default(view_93, [8, 128, 32, 32]);  view_93 = None
        mul_193 = torch.ops.aten.mul.Tensor(mul_192, expand_26);  mul_192 = expand_26 = None
        convolution_59 = torch.ops.aten.convolution.default(mul_193, arg85_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_193 = arg85_1 = None
        add_111 = torch.ops.aten.add.Tensor(arg87_1, 1e-05);  arg87_1 = None
        sqrt_47 = torch.ops.aten.sqrt.default(add_111);  add_111 = None
        reciprocal_47 = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_194 = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_376 = torch.ops.aten.unsqueeze.default(arg86_1, -1);  arg86_1 = None
        unsqueeze_377 = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        unsqueeze_378 = torch.ops.aten.unsqueeze.default(mul_194, -1);  mul_194 = None
        unsqueeze_379 = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        sub_50 = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_377);  convolution_59 = unsqueeze_377 = None
        mul_195 = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_379);  sub_50 = unsqueeze_379 = None
        unsqueeze_380 = torch.ops.aten.unsqueeze.default(arg88_1, -1);  arg88_1 = None
        unsqueeze_381 = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_196 = torch.ops.aten.mul.Tensor(mul_195, unsqueeze_381);  mul_195 = unsqueeze_381 = None
        unsqueeze_382 = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_383 = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_112 = torch.ops.aten.add.Tensor(mul_196, unsqueeze_383);  mul_196 = unsqueeze_383 = None
        add_113 = torch.ops.aten.add.Tensor(add_112, mul_184);  add_112 = mul_184 = None
        sigmoid_50 = torch.ops.aten.sigmoid.default(add_113)
        mul_197 = torch.ops.aten.mul.Tensor(add_113, sigmoid_50);  add_113 = sigmoid_50 = None
        convolution_60 = torch.ops.aten.convolution.default(mul_197, arg90_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg90_1 = None
        add_114 = torch.ops.aten.add.Tensor(arg92_1, 1e-05);  arg92_1 = None
        sqrt_48 = torch.ops.aten.sqrt.default(add_114);  add_114 = None
        reciprocal_48 = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_198 = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_384 = torch.ops.aten.unsqueeze.default(arg91_1, -1);  arg91_1 = None
        unsqueeze_385 = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        unsqueeze_386 = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_387 = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        sub_51 = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_385);  convolution_60 = unsqueeze_385 = None
        mul_199 = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_387);  sub_51 = unsqueeze_387 = None
        unsqueeze_388 = torch.ops.aten.unsqueeze.default(arg93_1, -1);  arg93_1 = None
        unsqueeze_389 = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_389);  mul_199 = unsqueeze_389 = None
        unsqueeze_390 = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_391 = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_115 = torch.ops.aten.add.Tensor(mul_200, unsqueeze_391);  mul_200 = unsqueeze_391 = None
        sigmoid_51 = torch.ops.aten.sigmoid.default(add_115)
        mul_201 = torch.ops.aten.mul.Tensor(add_115, sigmoid_51);  add_115 = sigmoid_51 = None
        convolution_61 = torch.ops.aten.convolution.default(mul_201, arg95_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 16);  mul_201 = arg95_1 = None
        add_116 = torch.ops.aten.add.Tensor(arg97_1, 1e-05);  arg97_1 = None
        sqrt_49 = torch.ops.aten.sqrt.default(add_116);  add_116 = None
        reciprocal_49 = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_202 = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_392 = torch.ops.aten.unsqueeze.default(arg96_1, -1);  arg96_1 = None
        unsqueeze_393 = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        unsqueeze_394 = torch.ops.aten.unsqueeze.default(mul_202, -1);  mul_202 = None
        unsqueeze_395 = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        sub_52 = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_393);  convolution_61 = unsqueeze_393 = None
        mul_203 = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_395);  sub_52 = unsqueeze_395 = None
        unsqueeze_396 = torch.ops.aten.unsqueeze.default(arg98_1, -1);  arg98_1 = None
        unsqueeze_397 = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_204 = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_397);  mul_203 = unsqueeze_397 = None
        unsqueeze_398 = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_399 = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_117 = torch.ops.aten.add.Tensor(mul_204, unsqueeze_399);  mul_204 = unsqueeze_399 = None
        sigmoid_52 = torch.ops.aten.sigmoid.default(add_117)
        mul_205 = torch.ops.aten.mul.Tensor(add_117, sigmoid_52);  add_117 = sigmoid_52 = None
        mean_10 = torch.ops.aten.mean.dim(mul_205, [2, 3])
        view_94 = torch.ops.aten.view.default(mean_10, [8, 1, -1]);  mean_10 = None
        convolution_62 = torch.ops.aten.convolution.default(view_94, arg100_1, None, [1], [2], [1], False, [0], 1);  view_94 = arg100_1 = None
        sigmoid_53 = torch.ops.aten.sigmoid.default(convolution_62);  convolution_62 = None
        view_95 = torch.ops.aten.view.default(sigmoid_53, [8, -1, 1, 1]);  sigmoid_53 = None
        expand_27 = torch.ops.aten.expand.default(view_95, [8, 256, 16, 16]);  view_95 = None
        mul_206 = torch.ops.aten.mul.Tensor(mul_205, expand_27);  mul_205 = expand_27 = None
        convolution_63 = torch.ops.aten.convolution.default(mul_206, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_206 = arg101_1 = None
        add_118 = torch.ops.aten.add.Tensor(arg103_1, 1e-05);  arg103_1 = None
        sqrt_50 = torch.ops.aten.sqrt.default(add_118);  add_118 = None
        reciprocal_50 = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_207 = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_400 = torch.ops.aten.unsqueeze.default(arg102_1, -1);  arg102_1 = None
        unsqueeze_401 = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        unsqueeze_402 = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_403 = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        sub_53 = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_401);  convolution_63 = unsqueeze_401 = None
        mul_208 = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_403);  sub_53 = unsqueeze_403 = None
        unsqueeze_404 = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_405 = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_405);  mul_208 = unsqueeze_405 = None
        unsqueeze_406 = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_407 = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_119 = torch.ops.aten.add.Tensor(mul_209, unsqueeze_407);  mul_209 = unsqueeze_407 = None
        convolution_64 = torch.ops.aten.convolution.default(mul_197, arg106_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_197 = arg106_1 = None
        add_120 = torch.ops.aten.add.Tensor(arg108_1, 1e-05);  arg108_1 = None
        sqrt_51 = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_51 = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_210 = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_408 = torch.ops.aten.unsqueeze.default(arg107_1, -1);  arg107_1 = None
        unsqueeze_409 = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        unsqueeze_410 = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
        unsqueeze_411 = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        sub_54 = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_409);  convolution_64 = unsqueeze_409 = None
        mul_211 = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_411);  sub_54 = unsqueeze_411 = None
        unsqueeze_412 = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_413 = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_413);  mul_211 = unsqueeze_413 = None
        unsqueeze_414 = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_415 = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_121 = torch.ops.aten.add.Tensor(mul_212, unsqueeze_415);  mul_212 = unsqueeze_415 = None
        add_122 = torch.ops.aten.add.Tensor(add_119, add_121);  add_119 = add_121 = None
        sigmoid_54 = torch.ops.aten.sigmoid.default(add_122)
        mul_213 = torch.ops.aten.mul.Tensor(add_122, sigmoid_54);  add_122 = sigmoid_54 = None
        convolution_65 = torch.ops.aten.convolution.default(mul_213, arg111_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg111_1 = None
        add_123 = torch.ops.aten.add.Tensor(arg113_1, 1e-05);  arg113_1 = None
        sqrt_52 = torch.ops.aten.sqrt.default(add_123);  add_123 = None
        reciprocal_52 = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_214 = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_416 = torch.ops.aten.unsqueeze.default(arg112_1, -1);  arg112_1 = None
        unsqueeze_417 = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        unsqueeze_418 = torch.ops.aten.unsqueeze.default(mul_214, -1);  mul_214 = None
        unsqueeze_419 = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        sub_55 = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_417);  convolution_65 = unsqueeze_417 = None
        mul_215 = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_419);  sub_55 = unsqueeze_419 = None
        unsqueeze_420 = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_421 = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_216 = torch.ops.aten.mul.Tensor(mul_215, unsqueeze_421);  mul_215 = unsqueeze_421 = None
        unsqueeze_422 = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_423 = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_124 = torch.ops.aten.add.Tensor(mul_216, unsqueeze_423);  mul_216 = unsqueeze_423 = None
        sigmoid_55 = torch.ops.aten.sigmoid.default(add_124)
        mul_217 = torch.ops.aten.mul.Tensor(add_124, sigmoid_55);  add_124 = sigmoid_55 = None
        convolution_66 = torch.ops.aten.convolution.default(mul_217, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg116_1 = None
        view_96 = torch.ops.aten.view.default(convolution_66, [-1, 16, 2, 8, 2, 8]);  convolution_66 = None
        permute_34 = torch.ops.aten.permute.default(view_96, [0, 1, 3, 5, 2, 4]);  view_96 = None
        clone_25 = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_97 = torch.ops.aten.view.default(clone_25, [64, 16, 64, 4]);  clone_25 = None
        permute_35 = torch.ops.aten.permute.default(view_97, [0, 3, 2, 1]);  view_97 = None
        convolution_67 = torch.ops.aten.convolution.default(mul_217, arg117_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_217 = arg117_1 = None
        constant_pad_nd_15 = torch.ops.aten.constant_pad_nd.default(convolution_67, [2, 2, 2, 2], 0.0);  convolution_67 = None
        unfold_6 = torch.ops.aten.unfold.default(constant_pad_nd_15, 2, 12, 8);  constant_pad_nd_15 = None
        unfold_7 = torch.ops.aten.unfold.default(unfold_6, 3, 12, 8);  unfold_6 = None
        clone_26 = torch.ops.aten.clone.default(unfold_7, memory_format = torch.contiguous_format);  unfold_7 = None
        view_98 = torch.ops.aten.view.default(clone_26, [64, 48, 4, 144]);  clone_26 = None
        permute_36 = torch.ops.aten.permute.default(view_98, [0, 2, 3, 1]);  view_98 = None
        split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(permute_36, [16, 32], -1);  permute_36 = None
        getitem_10 = split_with_sizes_3[0]
        getitem_11 = split_with_sizes_3[1];  split_with_sizes_3 = None
        permute_37 = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
        expand_28 = torch.ops.aten.expand.default(permute_35, [64, 4, 64, 16])
        clone_27 = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_99 = torch.ops.aten.view.default(clone_27, [256, 64, 16]);  clone_27 = None
        expand_29 = torch.ops.aten.expand.default(permute_37, [64, 4, 16, 144]);  permute_37 = None
        clone_28 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_100 = torch.ops.aten.view.default(clone_28, [256, 16, 144]);  clone_28 = None
        bmm_6 = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
        view_101 = torch.ops.aten.view.default(bmm_6, [64, 4, 64, 144]);  bmm_6 = None
        mul_218 = torch.ops.aten.mul.Tensor(view_101, 0.25);  view_101 = None
        clone_29 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_102 = torch.ops.aten.view.default(clone_29, [256, 8, 8, 16]);  clone_29 = None
        permute_38 = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        view_103 = torch.ops.aten.view.default(view_102, [16384, 16])
        mm_6 = torch.ops.aten.mm.default(view_103, permute_38);  view_103 = permute_38 = None
        view_104 = torch.ops.aten.view.default(mm_6, [256, 8, 8, 23]);  mm_6 = None
        view_105 = torch.ops.aten.view.default(view_104, [-1, 8, 23]);  view_104 = None
        constant_pad_nd_16 = torch.ops.aten.constant_pad_nd.default(view_105, [0, 1], 0.0);  view_105 = None
        view_106 = torch.ops.aten.view.default(constant_pad_nd_16, [2048, 192]);  constant_pad_nd_16 = None
        constant_pad_nd_17 = torch.ops.aten.constant_pad_nd.default(view_106, [0, 15], 0.0);  view_106 = None
        view_107 = torch.ops.aten.view.default(constant_pad_nd_17, [-1, 9, 23]);  constant_pad_nd_17 = None
        slice_20 = torch.ops.aten.slice.Tensor(view_107, 1, 0, 8);  view_107 = None
        slice_21 = torch.ops.aten.slice.Tensor(slice_20, 2, 11, 9223372036854775807);  slice_20 = None
        view_108 = torch.ops.aten.view.default(slice_21, [256, 8, 1, 8, 12]);  slice_21 = None
        expand_30 = torch.ops.aten.expand.default(view_108, [-1, -1, 12, -1, -1]);  view_108 = None
        permute_39 = torch.ops.aten.permute.default(expand_30, [0, 1, 3, 2, 4]);  expand_30 = None
        permute_40 = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        permute_41 = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        clone_30 = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_109 = torch.ops.aten.view.default(clone_30, [16384, 16]);  clone_30 = None
        mm_7 = torch.ops.aten.mm.default(view_109, permute_41);  view_109 = permute_41 = None
        view_110 = torch.ops.aten.view.default(mm_7, [256, 8, 8, 23]);  mm_7 = None
        view_111 = torch.ops.aten.view.default(view_110, [-1, 8, 23]);  view_110 = None
        constant_pad_nd_18 = torch.ops.aten.constant_pad_nd.default(view_111, [0, 1], 0.0);  view_111 = None
        view_112 = torch.ops.aten.view.default(constant_pad_nd_18, [2048, 192]);  constant_pad_nd_18 = None
        constant_pad_nd_19 = torch.ops.aten.constant_pad_nd.default(view_112, [0, 15], 0.0);  view_112 = None
        view_113 = torch.ops.aten.view.default(constant_pad_nd_19, [-1, 9, 23]);  constant_pad_nd_19 = None
        slice_23 = torch.ops.aten.slice.Tensor(view_113, 1, 0, 8);  view_113 = None
        slice_24 = torch.ops.aten.slice.Tensor(slice_23, 2, 11, 9223372036854775807);  slice_23 = None
        view_114 = torch.ops.aten.view.default(slice_24, [256, 8, 1, 8, 12]);  slice_24 = None
        expand_31 = torch.ops.aten.expand.default(view_114, [-1, -1, 12, -1, -1]);  view_114 = None
        permute_42 = torch.ops.aten.permute.default(expand_31, [0, 3, 1, 4, 2]);  expand_31 = None
        add_125 = torch.ops.aten.add.Tensor(permute_42, permute_39);  permute_42 = permute_39 = None
        clone_31 = torch.ops.aten.clone.default(add_125, memory_format = torch.contiguous_format);  add_125 = None
        view_115 = torch.ops.aten.view.default(clone_31, [64, 4, 64, 144]);  clone_31 = None
        add_126 = torch.ops.aten.add.Tensor(mul_218, view_115);  mul_218 = view_115 = None
        amax_3 = torch.ops.aten.amax.default(add_126, [-1], True)
        sub_56 = torch.ops.aten.sub.Tensor(add_126, amax_3);  add_126 = amax_3 = None
        exp_3 = torch.ops.aten.exp.default(sub_56);  sub_56 = None
        sum_4 = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3 = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        expand_32 = torch.ops.aten.expand.default(div_3, [64, 4, 64, 144]);  div_3 = None
        view_116 = torch.ops.aten.view.default(expand_32, [256, 64, 144]);  expand_32 = None
        expand_33 = torch.ops.aten.expand.default(getitem_11, [64, 4, 144, 32]);  getitem_11 = None
        clone_32 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_117 = torch.ops.aten.view.default(clone_32, [256, 144, 32]);  clone_32 = None
        bmm_7 = torch.ops.aten.bmm.default(view_116, view_117);  view_116 = view_117 = None
        view_118 = torch.ops.aten.view.default(bmm_7, [64, 4, 64, 32]);  bmm_7 = None
        permute_43 = torch.ops.aten.permute.default(view_118, [0, 3, 2, 1]);  view_118 = None
        clone_33 = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_119 = torch.ops.aten.view.default(clone_33, [2048, 8, 8, 2, 2]);  clone_33 = None
        permute_44 = torch.ops.aten.permute.default(view_119, [0, 3, 1, 4, 2]);  view_119 = None
        clone_34 = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_120 = torch.ops.aten.view.default(clone_34, [8, 256, 16, 16]);  clone_34 = None
        add_127 = torch.ops.aten.add.Tensor(arg121_1, 1e-05);  arg121_1 = None
        sqrt_53 = torch.ops.aten.sqrt.default(add_127);  add_127 = None
        reciprocal_53 = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_219 = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_424 = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_425 = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        unsqueeze_426 = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_427 = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        sub_57 = torch.ops.aten.sub.Tensor(view_120, unsqueeze_425);  view_120 = unsqueeze_425 = None
        mul_220 = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_427);  sub_57 = unsqueeze_427 = None
        unsqueeze_428 = torch.ops.aten.unsqueeze.default(arg122_1, -1);  arg122_1 = None
        unsqueeze_429 = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_429);  mul_220 = unsqueeze_429 = None
        unsqueeze_430 = torch.ops.aten.unsqueeze.default(arg123_1, -1);  arg123_1 = None
        unsqueeze_431 = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_128 = torch.ops.aten.add.Tensor(mul_221, unsqueeze_431);  mul_221 = unsqueeze_431 = None
        sigmoid_56 = torch.ops.aten.sigmoid.default(add_128)
        mul_222 = torch.ops.aten.mul.Tensor(add_128, sigmoid_56);  add_128 = sigmoid_56 = None
        convolution_68 = torch.ops.aten.convolution.default(mul_222, arg124_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_222 = arg124_1 = None
        add_129 = torch.ops.aten.add.Tensor(arg126_1, 1e-05);  arg126_1 = None
        sqrt_54 = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_54 = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_223 = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_432 = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_433 = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        unsqueeze_434 = torch.ops.aten.unsqueeze.default(mul_223, -1);  mul_223 = None
        unsqueeze_435 = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        sub_58 = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_433);  convolution_68 = unsqueeze_433 = None
        mul_224 = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_435);  sub_58 = unsqueeze_435 = None
        unsqueeze_436 = torch.ops.aten.unsqueeze.default(arg127_1, -1);  arg127_1 = None
        unsqueeze_437 = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_225 = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_437);  mul_224 = unsqueeze_437 = None
        unsqueeze_438 = torch.ops.aten.unsqueeze.default(arg128_1, -1);  arg128_1 = None
        unsqueeze_439 = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_130 = torch.ops.aten.add.Tensor(mul_225, unsqueeze_439);  mul_225 = unsqueeze_439 = None
        add_131 = torch.ops.aten.add.Tensor(add_130, mul_213);  add_130 = mul_213 = None
        sigmoid_57 = torch.ops.aten.sigmoid.default(add_131)
        mul_226 = torch.ops.aten.mul.Tensor(add_131, sigmoid_57);  add_131 = sigmoid_57 = None
        convolution_69 = torch.ops.aten.convolution.default(mul_226, arg129_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg129_1 = None
        add_132 = torch.ops.aten.add.Tensor(arg131_1, 1e-05);  arg131_1 = None
        sqrt_55 = torch.ops.aten.sqrt.default(add_132);  add_132 = None
        reciprocal_55 = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_227 = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_440 = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_441 = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        unsqueeze_442 = torch.ops.aten.unsqueeze.default(mul_227, -1);  mul_227 = None
        unsqueeze_443 = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        sub_59 = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_441);  convolution_69 = unsqueeze_441 = None
        mul_228 = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_443);  sub_59 = unsqueeze_443 = None
        unsqueeze_444 = torch.ops.aten.unsqueeze.default(arg132_1, -1);  arg132_1 = None
        unsqueeze_445 = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_229 = torch.ops.aten.mul.Tensor(mul_228, unsqueeze_445);  mul_228 = unsqueeze_445 = None
        unsqueeze_446 = torch.ops.aten.unsqueeze.default(arg133_1, -1);  arg133_1 = None
        unsqueeze_447 = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_133 = torch.ops.aten.add.Tensor(mul_229, unsqueeze_447);  mul_229 = unsqueeze_447 = None
        sigmoid_58 = torch.ops.aten.sigmoid.default(add_133)
        mul_230 = torch.ops.aten.mul.Tensor(add_133, sigmoid_58);  add_133 = sigmoid_58 = None
        convolution_70 = torch.ops.aten.convolution.default(mul_230, arg134_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  arg134_1 = None
        view_121 = torch.ops.aten.view.default(convolution_70, [-1, 16, 2, 4, 2, 4]);  convolution_70 = None
        permute_45 = torch.ops.aten.permute.default(view_121, [0, 1, 3, 5, 2, 4]);  view_121 = None
        clone_35 = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_122 = torch.ops.aten.view.default(clone_35, [64, 16, 16, 4]);  clone_35 = None
        permute_46 = torch.ops.aten.permute.default(view_122, [0, 3, 2, 1]);  view_122 = None
        convolution_71 = torch.ops.aten.convolution.default(mul_230, arg135_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_230 = arg135_1 = None
        constant_pad_nd_20 = torch.ops.aten.constant_pad_nd.default(convolution_71, [2, 2, 2, 2], 0.0);  convolution_71 = None
        unfold_8 = torch.ops.aten.unfold.default(constant_pad_nd_20, 2, 12, 8);  constant_pad_nd_20 = None
        unfold_9 = torch.ops.aten.unfold.default(unfold_8, 3, 12, 8);  unfold_8 = None
        clone_36 = torch.ops.aten.clone.default(unfold_9, memory_format = torch.contiguous_format);  unfold_9 = None
        view_123 = torch.ops.aten.view.default(clone_36, [64, 80, 4, 144]);  clone_36 = None
        permute_47 = torch.ops.aten.permute.default(view_123, [0, 2, 3, 1]);  view_123 = None
        split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(permute_47, [16, 64], -1);  permute_47 = None
        getitem_12 = split_with_sizes_4[0]
        getitem_13 = split_with_sizes_4[1];  split_with_sizes_4 = None
        permute_48 = torch.ops.aten.permute.default(getitem_12, [0, 1, 3, 2]);  getitem_12 = None
        expand_34 = torch.ops.aten.expand.default(permute_46, [64, 4, 16, 16])
        clone_37 = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_124 = torch.ops.aten.view.default(clone_37, [256, 16, 16]);  clone_37 = None
        expand_35 = torch.ops.aten.expand.default(permute_48, [64, 4, 16, 144]);  permute_48 = None
        clone_38 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_125 = torch.ops.aten.view.default(clone_38, [256, 16, 144]);  clone_38 = None
        bmm_8 = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
        view_126 = torch.ops.aten.view.default(bmm_8, [64, 4, 16, 144]);  bmm_8 = None
        mul_231 = torch.ops.aten.mul.Tensor(view_126, 0.25);  view_126 = None
        clone_39 = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_127 = torch.ops.aten.view.default(clone_39, [256, 4, 4, 16]);  clone_39 = None
        permute_49 = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
        view_128 = torch.ops.aten.view.default(view_127, [4096, 16])
        mm_8 = torch.ops.aten.mm.default(view_128, permute_49);  view_128 = permute_49 = None
        view_129 = torch.ops.aten.view.default(mm_8, [256, 4, 4, 23]);  mm_8 = None
        view_130 = torch.ops.aten.view.default(view_129, [-1, 4, 23]);  view_129 = None
        constant_pad_nd_21 = torch.ops.aten.constant_pad_nd.default(view_130, [0, 1], 0.0);  view_130 = None
        view_131 = torch.ops.aten.view.default(constant_pad_nd_21, [1024, 96]);  constant_pad_nd_21 = None
        constant_pad_nd_22 = torch.ops.aten.constant_pad_nd.default(view_131, [0, 19], 0.0);  view_131 = None
        view_132 = torch.ops.aten.view.default(constant_pad_nd_22, [-1, 5, 23]);  constant_pad_nd_22 = None
        slice_26 = torch.ops.aten.slice.Tensor(view_132, 1, 0, 4);  view_132 = None
        slice_27 = torch.ops.aten.slice.Tensor(slice_26, 2, 11, 9223372036854775807);  slice_26 = None
        view_133 = torch.ops.aten.view.default(slice_27, [256, 4, 1, 4, 12]);  slice_27 = None
        expand_36 = torch.ops.aten.expand.default(view_133, [-1, -1, 12, -1, -1]);  view_133 = None
        permute_50 = torch.ops.aten.permute.default(expand_36, [0, 1, 3, 2, 4]);  expand_36 = None
        permute_51 = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
        permute_52 = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        clone_40 = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_134 = torch.ops.aten.view.default(clone_40, [4096, 16]);  clone_40 = None
        mm_9 = torch.ops.aten.mm.default(view_134, permute_52);  view_134 = permute_52 = None
        view_135 = torch.ops.aten.view.default(mm_9, [256, 4, 4, 23]);  mm_9 = None
        view_136 = torch.ops.aten.view.default(view_135, [-1, 4, 23]);  view_135 = None
        constant_pad_nd_23 = torch.ops.aten.constant_pad_nd.default(view_136, [0, 1], 0.0);  view_136 = None
        view_137 = torch.ops.aten.view.default(constant_pad_nd_23, [1024, 96]);  constant_pad_nd_23 = None
        constant_pad_nd_24 = torch.ops.aten.constant_pad_nd.default(view_137, [0, 19], 0.0);  view_137 = None
        view_138 = torch.ops.aten.view.default(constant_pad_nd_24, [-1, 5, 23]);  constant_pad_nd_24 = None
        slice_29 = torch.ops.aten.slice.Tensor(view_138, 1, 0, 4);  view_138 = None
        slice_30 = torch.ops.aten.slice.Tensor(slice_29, 2, 11, 9223372036854775807);  slice_29 = None
        view_139 = torch.ops.aten.view.default(slice_30, [256, 4, 1, 4, 12]);  slice_30 = None
        expand_37 = torch.ops.aten.expand.default(view_139, [-1, -1, 12, -1, -1]);  view_139 = None
        permute_53 = torch.ops.aten.permute.default(expand_37, [0, 3, 1, 4, 2]);  expand_37 = None
        add_134 = torch.ops.aten.add.Tensor(permute_53, permute_50);  permute_53 = permute_50 = None
        clone_41 = torch.ops.aten.clone.default(add_134, memory_format = torch.contiguous_format);  add_134 = None
        view_140 = torch.ops.aten.view.default(clone_41, [64, 4, 16, 144]);  clone_41 = None
        add_135 = torch.ops.aten.add.Tensor(mul_231, view_140);  mul_231 = view_140 = None
        amax_4 = torch.ops.aten.amax.default(add_135, [-1], True)
        sub_60 = torch.ops.aten.sub.Tensor(add_135, amax_4);  add_135 = amax_4 = None
        exp_4 = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_5 = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4 = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        expand_38 = torch.ops.aten.expand.default(div_4, [64, 4, 16, 144]);  div_4 = None
        view_141 = torch.ops.aten.view.default(expand_38, [256, 16, 144]);  expand_38 = None
        expand_39 = torch.ops.aten.expand.default(getitem_13, [64, 4, 144, 64]);  getitem_13 = None
        clone_42 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_142 = torch.ops.aten.view.default(clone_42, [256, 144, 64]);  clone_42 = None
        bmm_9 = torch.ops.aten.bmm.default(view_141, view_142);  view_141 = view_142 = None
        view_143 = torch.ops.aten.view.default(bmm_9, [64, 4, 16, 64]);  bmm_9 = None
        permute_54 = torch.ops.aten.permute.default(view_143, [0, 3, 2, 1]);  view_143 = None
        clone_43 = torch.ops.aten.clone.default(permute_54, memory_format = torch.contiguous_format);  permute_54 = None
        view_144 = torch.ops.aten.view.default(clone_43, [4096, 4, 4, 2, 2]);  clone_43 = None
        permute_55 = torch.ops.aten.permute.default(view_144, [0, 3, 1, 4, 2]);  view_144 = None
        clone_44 = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_145 = torch.ops.aten.view.default(clone_44, [8, 512, 8, 8]);  clone_44 = None
        add_136 = torch.ops.aten.add.Tensor(arg139_1, 1e-05);  arg139_1 = None
        sqrt_56 = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_56 = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_232 = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_448 = torch.ops.aten.unsqueeze.default(arg138_1, -1);  arg138_1 = None
        unsqueeze_449 = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        unsqueeze_450 = torch.ops.aten.unsqueeze.default(mul_232, -1);  mul_232 = None
        unsqueeze_451 = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        sub_61 = torch.ops.aten.sub.Tensor(view_145, unsqueeze_449);  view_145 = unsqueeze_449 = None
        mul_233 = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_451);  sub_61 = unsqueeze_451 = None
        unsqueeze_452 = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_453 = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_234 = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_453);  mul_233 = unsqueeze_453 = None
        unsqueeze_454 = torch.ops.aten.unsqueeze.default(arg141_1, -1);  arg141_1 = None
        unsqueeze_455 = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_137 = torch.ops.aten.add.Tensor(mul_234, unsqueeze_455);  mul_234 = unsqueeze_455 = None
        sigmoid_59 = torch.ops.aten.sigmoid.default(add_137)
        mul_235 = torch.ops.aten.mul.Tensor(add_137, sigmoid_59);  add_137 = sigmoid_59 = None
        convolution_72 = torch.ops.aten.convolution.default(mul_235, arg142_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_235 = arg142_1 = None
        add_138 = torch.ops.aten.add.Tensor(arg144_1, 1e-05);  arg144_1 = None
        sqrt_57 = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_57 = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_236 = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_456 = torch.ops.aten.unsqueeze.default(arg143_1, -1);  arg143_1 = None
        unsqueeze_457 = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        unsqueeze_458 = torch.ops.aten.unsqueeze.default(mul_236, -1);  mul_236 = None
        unsqueeze_459 = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        sub_62 = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_457);  convolution_72 = unsqueeze_457 = None
        mul_237 = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_459);  sub_62 = unsqueeze_459 = None
        unsqueeze_460 = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_461 = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_238 = torch.ops.aten.mul.Tensor(mul_237, unsqueeze_461);  mul_237 = unsqueeze_461 = None
        unsqueeze_462 = torch.ops.aten.unsqueeze.default(arg146_1, -1);  arg146_1 = None
        unsqueeze_463 = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_139 = torch.ops.aten.add.Tensor(mul_238, unsqueeze_463);  mul_238 = unsqueeze_463 = None
        convolution_73 = torch.ops.aten.convolution.default(mul_226, arg147_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  mul_226 = arg147_1 = None
        add_140 = torch.ops.aten.add.Tensor(arg149_1, 1e-05);  arg149_1 = None
        sqrt_58 = torch.ops.aten.sqrt.default(add_140);  add_140 = None
        reciprocal_58 = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_239 = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_464 = torch.ops.aten.unsqueeze.default(arg148_1, -1);  arg148_1 = None
        unsqueeze_465 = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        unsqueeze_466 = torch.ops.aten.unsqueeze.default(mul_239, -1);  mul_239 = None
        unsqueeze_467 = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        sub_63 = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_465);  convolution_73 = unsqueeze_465 = None
        mul_240 = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_467);  sub_63 = unsqueeze_467 = None
        unsqueeze_468 = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_469 = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_241 = torch.ops.aten.mul.Tensor(mul_240, unsqueeze_469);  mul_240 = unsqueeze_469 = None
        unsqueeze_470 = torch.ops.aten.unsqueeze.default(arg151_1, -1);  arg151_1 = None
        unsqueeze_471 = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_141 = torch.ops.aten.add.Tensor(mul_241, unsqueeze_471);  mul_241 = unsqueeze_471 = None
        add_142 = torch.ops.aten.add.Tensor(add_139, add_141);  add_139 = add_141 = None
        sigmoid_60 = torch.ops.aten.sigmoid.default(add_142)
        mul_242 = torch.ops.aten.mul.Tensor(add_142, sigmoid_60);  add_142 = sigmoid_60 = None
        convolution_74 = torch.ops.aten.convolution.default(mul_242, arg152_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg152_1 = None
        add_143 = torch.ops.aten.add.Tensor(arg154_1, 1e-05);  arg154_1 = None
        sqrt_59 = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_59 = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_243 = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_472 = torch.ops.aten.unsqueeze.default(arg153_1, -1);  arg153_1 = None
        unsqueeze_473 = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        unsqueeze_474 = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_475 = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        sub_64 = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_473);  convolution_74 = unsqueeze_473 = None
        mul_244 = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_475);  sub_64 = unsqueeze_475 = None
        unsqueeze_476 = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_477 = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_477);  mul_244 = unsqueeze_477 = None
        unsqueeze_478 = torch.ops.aten.unsqueeze.default(arg156_1, -1);  arg156_1 = None
        unsqueeze_479 = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_144 = torch.ops.aten.add.Tensor(mul_245, unsqueeze_479);  mul_245 = unsqueeze_479 = None
        sigmoid_61 = torch.ops.aten.sigmoid.default(add_144)
        mul_246 = torch.ops.aten.mul.Tensor(add_144, sigmoid_61);  add_144 = sigmoid_61 = None
        convolution_75 = torch.ops.aten.convolution.default(mul_246, arg157_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg157_1 = None
        view_146 = torch.ops.aten.view.default(convolution_75, [-1, 16, 1, 8, 1, 8]);  convolution_75 = None
        permute_56 = torch.ops.aten.permute.default(view_146, [0, 1, 3, 5, 2, 4]);  view_146 = None
        view_147 = torch.ops.aten.view.default(permute_56, [64, 16, -1, 1]);  permute_56 = None
        permute_57 = torch.ops.aten.permute.default(view_147, [0, 3, 2, 1]);  view_147 = None
        convolution_76 = torch.ops.aten.convolution.default(mul_246, arg158_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_246 = arg158_1 = None
        constant_pad_nd_25 = torch.ops.aten.constant_pad_nd.default(convolution_76, [2, 2, 2, 2], 0.0);  convolution_76 = None
        unfold_10 = torch.ops.aten.unfold.default(constant_pad_nd_25, 2, 12, 8);  constant_pad_nd_25 = None
        unfold_11 = torch.ops.aten.unfold.default(unfold_10, 3, 12, 8);  unfold_10 = None
        view_148 = torch.ops.aten.view.default(unfold_11, [64, 80, 1, -1]);  unfold_11 = None
        permute_58 = torch.ops.aten.permute.default(view_148, [0, 2, 3, 1]);  view_148 = None
        split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(permute_58, [16, 64], -1);  permute_58 = None
        getitem_14 = split_with_sizes_5[0]
        getitem_15 = split_with_sizes_5[1];  split_with_sizes_5 = None
        permute_59 = torch.ops.aten.permute.default(getitem_14, [0, 1, 3, 2]);  getitem_14 = None
        expand_40 = torch.ops.aten.expand.default(permute_57, [64, 1, 64, 16])
        view_149 = torch.ops.aten.view.default(expand_40, [64, 64, 16]);  expand_40 = None
        expand_41 = torch.ops.aten.expand.default(permute_59, [64, 1, 16, 144]);  permute_59 = None
        view_150 = torch.ops.aten.view.default(expand_41, [64, 16, 144]);  expand_41 = None
        bmm_10 = torch.ops.aten.bmm.default(view_149, view_150);  view_149 = view_150 = None
        view_151 = torch.ops.aten.view.default(bmm_10, [64, 1, 64, 144]);  bmm_10 = None
        mul_247 = torch.ops.aten.mul.Tensor(view_151, 0.25);  view_151 = None
        view_152 = torch.ops.aten.view.default(permute_57, [64, 8, 8, 16]);  permute_57 = None
        permute_60 = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        clone_45 = torch.ops.aten.clone.default(view_152, memory_format = torch.contiguous_format)
        view_153 = torch.ops.aten.view.default(clone_45, [4096, 16]);  clone_45 = None
        mm_10 = torch.ops.aten.mm.default(view_153, permute_60);  view_153 = permute_60 = None
        view_154 = torch.ops.aten.view.default(mm_10, [64, 8, 8, 23]);  mm_10 = None
        view_155 = torch.ops.aten.view.default(view_154, [-1, 8, 23]);  view_154 = None
        constant_pad_nd_26 = torch.ops.aten.constant_pad_nd.default(view_155, [0, 1], 0.0);  view_155 = None
        view_156 = torch.ops.aten.view.default(constant_pad_nd_26, [512, 192]);  constant_pad_nd_26 = None
        constant_pad_nd_27 = torch.ops.aten.constant_pad_nd.default(view_156, [0, 15], 0.0);  view_156 = None
        view_157 = torch.ops.aten.view.default(constant_pad_nd_27, [-1, 9, 23]);  constant_pad_nd_27 = None
        slice_32 = torch.ops.aten.slice.Tensor(view_157, 1, 0, 8);  view_157 = None
        slice_33 = torch.ops.aten.slice.Tensor(slice_32, 2, 11, 9223372036854775807);  slice_32 = None
        view_158 = torch.ops.aten.view.default(slice_33, [64, 8, 1, 8, 12]);  slice_33 = None
        expand_42 = torch.ops.aten.expand.default(view_158, [-1, -1, 12, -1, -1]);  view_158 = None
        permute_61 = torch.ops.aten.permute.default(expand_42, [0, 1, 3, 2, 4]);  expand_42 = None
        permute_62 = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        permute_63 = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
        clone_46 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_159 = torch.ops.aten.view.default(clone_46, [4096, 16]);  clone_46 = None
        mm_11 = torch.ops.aten.mm.default(view_159, permute_63);  view_159 = permute_63 = None
        view_160 = torch.ops.aten.view.default(mm_11, [64, 8, 8, 23]);  mm_11 = None
        view_161 = torch.ops.aten.view.default(view_160, [-1, 8, 23]);  view_160 = None
        constant_pad_nd_28 = torch.ops.aten.constant_pad_nd.default(view_161, [0, 1], 0.0);  view_161 = None
        view_162 = torch.ops.aten.view.default(constant_pad_nd_28, [512, 192]);  constant_pad_nd_28 = None
        constant_pad_nd_29 = torch.ops.aten.constant_pad_nd.default(view_162, [0, 15], 0.0);  view_162 = None
        view_163 = torch.ops.aten.view.default(constant_pad_nd_29, [-1, 9, 23]);  constant_pad_nd_29 = None
        slice_35 = torch.ops.aten.slice.Tensor(view_163, 1, 0, 8);  view_163 = None
        slice_36 = torch.ops.aten.slice.Tensor(slice_35, 2, 11, 9223372036854775807);  slice_35 = None
        view_164 = torch.ops.aten.view.default(slice_36, [64, 8, 1, 8, 12]);  slice_36 = None
        expand_43 = torch.ops.aten.expand.default(view_164, [-1, -1, 12, -1, -1]);  view_164 = None
        permute_64 = torch.ops.aten.permute.default(expand_43, [0, 3, 1, 4, 2]);  expand_43 = None
        add_145 = torch.ops.aten.add.Tensor(permute_64, permute_61);  permute_64 = permute_61 = None
        clone_47 = torch.ops.aten.clone.default(add_145, memory_format = torch.contiguous_format);  add_145 = None
        view_165 = torch.ops.aten.view.default(clone_47, [64, 1, 64, 144]);  clone_47 = None
        add_146 = torch.ops.aten.add.Tensor(mul_247, view_165);  mul_247 = view_165 = None
        amax_5 = torch.ops.aten.amax.default(add_146, [-1], True)
        sub_65 = torch.ops.aten.sub.Tensor(add_146, amax_5);  add_146 = amax_5 = None
        exp_5 = torch.ops.aten.exp.default(sub_65);  sub_65 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5 = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        expand_44 = torch.ops.aten.expand.default(div_5, [64, 1, 64, 144]);  div_5 = None
        view_166 = torch.ops.aten.view.default(expand_44, [64, 64, 144]);  expand_44 = None
        expand_45 = torch.ops.aten.expand.default(getitem_15, [64, 1, 144, 64]);  getitem_15 = None
        view_167 = torch.ops.aten.view.default(expand_45, [64, 144, 64]);  expand_45 = None
        bmm_11 = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
        view_168 = torch.ops.aten.view.default(bmm_11, [64, 1, 64, 64]);  bmm_11 = None
        permute_65 = torch.ops.aten.permute.default(view_168, [0, 3, 2, 1]);  view_168 = None
        clone_48 = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
        view_169 = torch.ops.aten.view.default(clone_48, [4096, 8, 8, 1, 1]);  clone_48 = None
        permute_66 = torch.ops.aten.permute.default(view_169, [0, 3, 1, 4, 2]);  view_169 = None
        view_170 = torch.ops.aten.view.default(permute_66, [8, 512, 8, 8]);  permute_66 = None
        add_147 = torch.ops.aten.add.Tensor(arg162_1, 1e-05);  arg162_1 = None
        sqrt_60 = torch.ops.aten.sqrt.default(add_147);  add_147 = None
        reciprocal_60 = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_248 = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_480 = torch.ops.aten.unsqueeze.default(arg161_1, -1);  arg161_1 = None
        unsqueeze_481 = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        unsqueeze_482 = torch.ops.aten.unsqueeze.default(mul_248, -1);  mul_248 = None
        unsqueeze_483 = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        sub_66 = torch.ops.aten.sub.Tensor(view_170, unsqueeze_481);  view_170 = unsqueeze_481 = None
        mul_249 = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_483);  sub_66 = unsqueeze_483 = None
        unsqueeze_484 = torch.ops.aten.unsqueeze.default(arg163_1, -1);  arg163_1 = None
        unsqueeze_485 = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_250 = torch.ops.aten.mul.Tensor(mul_249, unsqueeze_485);  mul_249 = unsqueeze_485 = None
        unsqueeze_486 = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_487 = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_148 = torch.ops.aten.add.Tensor(mul_250, unsqueeze_487);  mul_250 = unsqueeze_487 = None
        sigmoid_62 = torch.ops.aten.sigmoid.default(add_148)
        mul_251 = torch.ops.aten.mul.Tensor(add_148, sigmoid_62);  add_148 = sigmoid_62 = None
        convolution_77 = torch.ops.aten.convolution.default(mul_251, arg165_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  mul_251 = arg165_1 = None
        add_149 = torch.ops.aten.add.Tensor(arg167_1, 1e-05);  arg167_1 = None
        sqrt_61 = torch.ops.aten.sqrt.default(add_149);  add_149 = None
        reciprocal_61 = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_252 = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_488 = torch.ops.aten.unsqueeze.default(arg166_1, -1);  arg166_1 = None
        unsqueeze_489 = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        unsqueeze_490 = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
        unsqueeze_491 = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        sub_67 = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_489);  convolution_77 = unsqueeze_489 = None
        mul_253 = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_491);  sub_67 = unsqueeze_491 = None
        unsqueeze_492 = torch.ops.aten.unsqueeze.default(arg168_1, -1);  arg168_1 = None
        unsqueeze_493 = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_254 = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_493);  mul_253 = unsqueeze_493 = None
        unsqueeze_494 = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_495 = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_150 = torch.ops.aten.add.Tensor(mul_254, unsqueeze_495);  mul_254 = unsqueeze_495 = None
        add_151 = torch.ops.aten.add.Tensor(add_150, mul_242);  add_150 = mul_242 = None
        sigmoid_63 = torch.ops.aten.sigmoid.default(add_151)
        mul_255 = torch.ops.aten.mul.Tensor(add_151, sigmoid_63);  add_151 = sigmoid_63 = None
        mean_11 = torch.ops.aten.mean.dim(mul_255, [-1, -2], True);  mul_255 = None
        view_171 = torch.ops.aten.view.default(mean_11, [8, 2048]);  mean_11 = None
        permute_67 = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
        addmm_1 = torch.ops.aten.addmm.default(arg171_1, view_171, permute_67);  arg171_1 = view_171 = permute_67 = None
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
    buf21 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf21, (64, 16, 3, 3), is_leaf=True)  # arg21_1
    buf22 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf22, (64,), is_leaf=True)  # arg22_1
    buf23 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf23, (64,), is_leaf=True)  # arg23_1
    buf24 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf24, (64,), is_leaf=True)  # arg24_1
    buf25 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf25, (64,), is_leaf=True)  # arg25_1
    buf26 = reader.storage(None, 12, device=device(type='cuda', index=0))
    reader.tensor(buf26, (1, 1, 3), is_leaf=True)  # arg26_1
    buf27 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf27, (256, 64, 1, 1), is_leaf=True)  # arg27_1
    buf28 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf28, (256,), is_leaf=True)  # arg28_1
    buf29 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf29, (256,), is_leaf=True)  # arg29_1
    buf30 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256,), is_leaf=True)  # arg30_1
    buf31 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256,), is_leaf=True)  # arg31_1
    buf32 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf32, (256, 64, 1, 1), is_leaf=True)  # arg32_1
    buf33 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256,), is_leaf=True)  # arg33_1
    buf34 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), is_leaf=True)  # arg34_1
    buf35 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256,), is_leaf=True)  # arg35_1
    buf36 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf36, (256,), is_leaf=True)  # arg36_1
    buf37 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf37, (64, 256, 1, 1), is_leaf=True)  # arg37_1
    buf38 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf38, (64,), is_leaf=True)  # arg38_1
    buf39 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf39, (64,), is_leaf=True)  # arg39_1
    buf40 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf40, (64,), is_leaf=True)  # arg40_1
    buf41 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf41, (64,), is_leaf=True)  # arg41_1
    buf42 = reader.storage(None, 36864, device=device(type='cuda', index=0))
    reader.tensor(buf42, (64, 16, 3, 3), is_leaf=True)  # arg42_1
    buf43 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf43, (64,), is_leaf=True)  # arg43_1
    buf44 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf44, (64,), is_leaf=True)  # arg44_1
    buf45 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf45, (64,), is_leaf=True)  # arg45_1
    buf46 = reader.storage(None, 256, device=device(type='cuda', index=0))
    reader.tensor(buf46, (64,), is_leaf=True)  # arg46_1
    buf47 = reader.storage(None, 12, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1, 1, 3), is_leaf=True)  # arg47_1
    buf48 = reader.storage(None, 65536, device=device(type='cuda', index=0))
    reader.tensor(buf48, (256, 64, 1, 1), is_leaf=True)  # arg48_1
    buf49 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf49, (256,), is_leaf=True)  # arg49_1
    buf50 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf50, (256,), is_leaf=True)  # arg50_1
    buf51 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf51, (256,), is_leaf=True)  # arg51_1
    buf52 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf52, (256,), is_leaf=True)  # arg52_1
    buf53 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf53, (128, 256, 1, 1), is_leaf=True)  # arg53_1
    buf54 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf54, (128,), is_leaf=True)  # arg54_1
    buf55 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf55, (128,), is_leaf=True)  # arg55_1
    buf56 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf56, (128,), is_leaf=True)  # arg56_1
    buf57 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf57, (128,), is_leaf=True)  # arg57_1
    buf58 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf58, (128, 16, 3, 3), is_leaf=True)  # arg58_1
    buf59 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf59, (128,), is_leaf=True)  # arg59_1
    buf60 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf60, (128,), is_leaf=True)  # arg60_1
    buf61 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf61, (128,), is_leaf=True)  # arg61_1
    buf62 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf62, (128,), is_leaf=True)  # arg62_1
    buf63 = reader.storage(None, 20, device=device(type='cuda', index=0))
    reader.tensor(buf63, (1, 1, 5), is_leaf=True)  # arg63_1
    buf64 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf64, (512, 128, 1, 1), is_leaf=True)  # arg64_1
    buf65 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf65, (512,), is_leaf=True)  # arg65_1
    buf66 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf66, (512,), is_leaf=True)  # arg66_1
    buf67 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf67, (512,), is_leaf=True)  # arg67_1
    buf68 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf68, (512,), is_leaf=True)  # arg68_1
    buf69 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf69, (512, 256, 1, 1), is_leaf=True)  # arg69_1
    buf70 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf70, (512,), is_leaf=True)  # arg70_1
    buf71 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf71, (512,), is_leaf=True)  # arg71_1
    buf72 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf72, (512,), is_leaf=True)  # arg72_1
    buf73 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf73, (512,), is_leaf=True)  # arg73_1
    buf74 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf74, (128, 512, 1, 1), is_leaf=True)  # arg74_1
    buf75 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf75, (128,), is_leaf=True)  # arg75_1
    buf76 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf76, (128,), is_leaf=True)  # arg76_1
    buf77 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf77, (128,), is_leaf=True)  # arg77_1
    buf78 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf78, (128,), is_leaf=True)  # arg78_1
    buf79 = reader.storage(None, 73728, device=device(type='cuda', index=0))
    reader.tensor(buf79, (128, 16, 3, 3), is_leaf=True)  # arg79_1
    buf80 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf80, (128,), is_leaf=True)  # arg80_1
    buf81 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf81, (128,), is_leaf=True)  # arg81_1
    buf82 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf82, (128,), is_leaf=True)  # arg82_1
    buf83 = reader.storage(None, 512, device=device(type='cuda', index=0))
    reader.tensor(buf83, (128,), is_leaf=True)  # arg83_1
    buf84 = reader.storage(None, 20, device=device(type='cuda', index=0))
    reader.tensor(buf84, (1, 1, 5), is_leaf=True)  # arg84_1
    buf85 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf85, (512, 128, 1, 1), is_leaf=True)  # arg85_1
    buf86 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf86, (512,), is_leaf=True)  # arg86_1
    buf87 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf87, (512,), is_leaf=True)  # arg87_1
    buf88 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf88, (512,), is_leaf=True)  # arg88_1
    buf89 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf89, (512,), is_leaf=True)  # arg89_1
    buf90 = reader.storage(None, 524288, device=device(type='cuda', index=0))
    reader.tensor(buf90, (256, 512, 1, 1), is_leaf=True)  # arg90_1
    buf91 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf91, (256,), is_leaf=True)  # arg91_1
    buf92 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf92, (256,), is_leaf=True)  # arg92_1
    buf93 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf93, (256,), is_leaf=True)  # arg93_1
    buf94 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf94, (256,), is_leaf=True)  # arg94_1
    buf95 = reader.storage(None, 147456, device=device(type='cuda', index=0))
    reader.tensor(buf95, (256, 16, 3, 3), is_leaf=True)  # arg95_1
    buf96 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf96, (256,), is_leaf=True)  # arg96_1
    buf97 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf97, (256,), is_leaf=True)  # arg97_1
    buf98 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf98, (256,), is_leaf=True)  # arg98_1
    buf99 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf99, (256,), is_leaf=True)  # arg99_1
    buf100 = reader.storage(None, 20, device=device(type='cuda', index=0))
    reader.tensor(buf100, (1, 1, 5), is_leaf=True)  # arg100_1
    buf101 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf101, (1024, 256, 1, 1), is_leaf=True)  # arg101_1
    buf102 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf102, (1024,), is_leaf=True)  # arg102_1
    buf103 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf103, (1024,), is_leaf=True)  # arg103_1
    buf104 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf104, (1024,), is_leaf=True)  # arg104_1
    buf105 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf105, (1024,), is_leaf=True)  # arg105_1
    buf106 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf106, (1024, 512, 1, 1), is_leaf=True)  # arg106_1
    buf107 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf107, (1024,), is_leaf=True)  # arg107_1
    buf108 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf108, (1024,), is_leaf=True)  # arg108_1
    buf109 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf109, (1024,), is_leaf=True)  # arg109_1
    buf110 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf110, (1024,), is_leaf=True)  # arg110_1
    buf111 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf111, (256, 1024, 1, 1), is_leaf=True)  # arg111_1
    buf112 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf112, (256,), is_leaf=True)  # arg112_1
    buf113 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf113, (256,), is_leaf=True)  # arg113_1
    buf114 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf114, (256,), is_leaf=True)  # arg114_1
    buf115 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf115, (256,), is_leaf=True)  # arg115_1
    buf116 = reader.storage(None, 131072, device=device(type='cuda', index=0))
    reader.tensor(buf116, (128, 256, 1, 1), is_leaf=True)  # arg116_1
    buf117 = reader.storage(None, 393216, device=device(type='cuda', index=0))
    reader.tensor(buf117, (384, 256, 1, 1), is_leaf=True)  # arg117_1
    buf118 = reader.storage(None, 1472, device=device(type='cuda', index=0))
    reader.tensor(buf118, (23, 16), is_leaf=True)  # arg118_1
    buf119 = reader.storage(None, 1472, device=device(type='cuda', index=0))
    reader.tensor(buf119, (23, 16), is_leaf=True)  # arg119_1
    buf120 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf120, (256,), is_leaf=True)  # arg120_1
    buf121 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf121, (256,), is_leaf=True)  # arg121_1
    buf122 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf122, (256,), is_leaf=True)  # arg122_1
    buf123 = reader.storage(None, 1024, device=device(type='cuda', index=0))
    reader.tensor(buf123, (256,), is_leaf=True)  # arg123_1
    buf124 = reader.storage(None, 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf124, (1024, 256, 1, 1), is_leaf=True)  # arg124_1
    buf125 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf125, (1024,), is_leaf=True)  # arg125_1
    buf126 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf126, (1024,), is_leaf=True)  # arg126_1
    buf127 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf127, (1024,), is_leaf=True)  # arg127_1
    buf128 = reader.storage(None, 4096, device=device(type='cuda', index=0))
    reader.tensor(buf128, (1024,), is_leaf=True)  # arg128_1
    buf129 = reader.storage(None, 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf129, (512, 1024, 1, 1), is_leaf=True)  # arg129_1
    buf130 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf130, (512,), is_leaf=True)  # arg130_1
    buf131 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf131, (512,), is_leaf=True)  # arg131_1
    buf132 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf132, (512,), is_leaf=True)  # arg132_1
    buf133 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf133, (512,), is_leaf=True)  # arg133_1
    buf134 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf134, (128, 512, 1, 1), is_leaf=True)  # arg134_1
    buf135 = reader.storage(None, 1310720, device=device(type='cuda', index=0))
    reader.tensor(buf135, (640, 512, 1, 1), is_leaf=True)  # arg135_1
    buf136 = reader.storage(None, 1472, device=device(type='cuda', index=0))
    reader.tensor(buf136, (23, 16), is_leaf=True)  # arg136_1
    buf137 = reader.storage(None, 1472, device=device(type='cuda', index=0))
    reader.tensor(buf137, (23, 16), is_leaf=True)  # arg137_1
    buf138 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf138, (512,), is_leaf=True)  # arg138_1
    buf139 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf139, (512,), is_leaf=True)  # arg139_1
    buf140 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf140, (512,), is_leaf=True)  # arg140_1
    buf141 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf141, (512,), is_leaf=True)  # arg141_1
    buf142 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf142, (2048, 512, 1, 1), is_leaf=True)  # arg142_1
    buf143 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf143, (2048,), is_leaf=True)  # arg143_1
    buf144 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf144, (2048,), is_leaf=True)  # arg144_1
    buf145 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf145, (2048,), is_leaf=True)  # arg145_1
    buf146 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf146, (2048,), is_leaf=True)  # arg146_1
    buf147 = reader.storage(None, 8388608, device=device(type='cuda', index=0))
    reader.tensor(buf147, (2048, 1024, 1, 1), is_leaf=True)  # arg147_1
    buf148 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf148, (2048,), is_leaf=True)  # arg148_1
    buf149 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf149, (2048,), is_leaf=True)  # arg149_1
    buf150 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf150, (2048,), is_leaf=True)  # arg150_1
    buf151 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf151, (2048,), is_leaf=True)  # arg151_1
    buf152 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf152, (512, 2048, 1, 1), is_leaf=True)  # arg152_1
    buf153 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf153, (512,), is_leaf=True)  # arg153_1
    buf154 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf154, (512,), is_leaf=True)  # arg154_1
    buf155 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf155, (512,), is_leaf=True)  # arg155_1
    buf156 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf156, (512,), is_leaf=True)  # arg156_1
    buf157 = reader.storage(None, 262144, device=device(type='cuda', index=0))
    reader.tensor(buf157, (128, 512, 1, 1), is_leaf=True)  # arg157_1
    buf158 = reader.storage(None, 1310720, device=device(type='cuda', index=0))
    reader.tensor(buf158, (640, 512, 1, 1), is_leaf=True)  # arg158_1
    buf159 = reader.storage(None, 1472, device=device(type='cuda', index=0))
    reader.tensor(buf159, (23, 16), is_leaf=True)  # arg159_1
    buf160 = reader.storage(None, 1472, device=device(type='cuda', index=0))
    reader.tensor(buf160, (23, 16), is_leaf=True)  # arg160_1
    buf161 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf161, (512,), is_leaf=True)  # arg161_1
    buf162 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf162, (512,), is_leaf=True)  # arg162_1
    buf163 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf163, (512,), is_leaf=True)  # arg163_1
    buf164 = reader.storage(None, 2048, device=device(type='cuda', index=0))
    reader.tensor(buf164, (512,), is_leaf=True)  # arg164_1
    buf165 = reader.storage(None, 4194304, device=device(type='cuda', index=0))
    reader.tensor(buf165, (2048, 512, 1, 1), is_leaf=True)  # arg165_1
    buf166 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf166, (2048,), is_leaf=True)  # arg166_1
    buf167 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf167, (2048,), is_leaf=True)  # arg167_1
    buf168 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf168, (2048,), is_leaf=True)  # arg168_1
    buf169 = reader.storage(None, 8192, device=device(type='cuda', index=0))
    reader.tensor(buf169, (2048,), is_leaf=True)  # arg169_1
    buf170 = reader.storage(None, 8192000, device=device(type='cuda', index=0))
    reader.tensor(buf170, (1000, 2048), is_leaf=True)  # arg170_1
    buf171 = reader.storage(None, 4000, device=device(type='cuda', index=0))
    reader.tensor(buf171, (1000,), is_leaf=True)  # arg171_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)